import argparse
import json
import sys
import os
import time
import json as _json
import difflib  # DiffChecker dependency

# Ensure current directory (src/) is on sys.path so local modules can be imported
sys.path.append(".")
sys.path.append(os.path.dirname(__file__))

from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import set_seed
import torch

from utils import boolean_string
from utils import get_current_time
from verify_fix import verify_candidate

# --- DiffChecker: Color Constants ---
GREEN = '\033[32m'
RED = '\033[91m'
RESET = '\033[0m'
CYAN = '\033[96m'


def generate_colored_diff(original: str, fixed: str, use_color: bool = True) -> str:
    """
    Generates a unified diff between original code and the predicted fix.
    """
    original_lines = original.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)

    # Generate unified diff with 3 lines of context
    diff = difflib.unified_diff(
        original_lines,
        fixed_lines,
        fromfile="Original",
        tofile="Prediction",
        n=3,
    )

    output = []
    for line in diff:
        if use_color:
            if line.startswith("+") and not line.startswith("+++"):
                output.append(GREEN + line.strip() + RESET)
            elif line.startswith("-") and not line.startswith("---"):
                output.append(RED + line.strip() + RESET)
            elif line.startswith("^"):
                output.append(line.strip())
            else:
                output.append(line.strip())
        else:
            output.append(line.strip())

    return "\n".join(output)


def get_single_prediction(model, tokenizer, input_text, max_length=256, beam_size=50, num_seq=50):
    """
    Generate one or more candidate fixes for the given input text.
    """
    # Tokenize the input text
    # Note: .to(model.device) ensures inputs go to the same hardware (CPU or GPU) as the model
    input_ids = tokenizer.encode(
        input_text, truncation=True, padding=True, return_tensors="pt"
    ).to(model.device)
    # Generate predictions
    beam_outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=beam_size,
        num_return_sequences=num_seq,
        early_stopping=False,
    )
    # Decode the predictions
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in beam_outputs]

    return predictions


# transformers.logging.set_verbosity_info()
set_seed(42)
print("start time: ", get_current_time())

parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batch-size", type=int, default=1)
parser.add_argument(
    "-mn",
    "--model-name",
    type=str,
    # choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
    required=True,
)
parser.add_argument(
    "-lm",
    "--load-model",
    type=str,
    default="",
)  #  Checkpoint dir to load the model. Example: t5-small_global_14-12-2020_16-29-22/checkpoint-10
parser.add_argument(
    "-ea",
    "--eval-all",
    type=boolean_string,
    default=False,
)  # to evaluate on all data or not
parser.add_argument("-eas", "--eval-acc-steps", type=int, default=1)
# parser.add_argument("-md", "--model-dir", type=str, default="")
parser.add_argument("-et", "--error-type", type=str, default="")
parser.add_argument("-bm", "--beam-size", type=int, default=50)  # number of beams to use
parser.add_argument(
    "-seq",
    "--num-seq",
    type=int,
    default=50,
)  # number of seq to generate, must be <= number of beams
parser.add_argument(
    "-f",
    "--file_path",
    type=str,
    required=True,
    help="Enter the path to the file containing input.",
)
parser.add_argument(
    "-vf",
    "--verify",
    type=boolean_string,
    default=True,
    help="Whether to run static-check verification (pyre/mypy) on each candidate",
)

# --- DiffChecker: New Arguments ---
parser.add_argument(
    "--diff",
    action="store_true",
    help="Enable enhanced diff output (DiffChecker)",
)
parser.add_argument(
    "--save-diff",
    action="store_true",
    help="Save the diffs to a file named pyty_suggestions.diff",
)
parser.add_argument(
    "--top",
    type=int,
    default=50,
    help="Limit the number of suggestions to display/save",
)

args = parser.parse_args()

model_name = args.model_name

# Validate model path before attempting to load locally
if args.load_model:
    if not os.path.exists(args.load_model):
        print("ERROR: The provided --load-model path does not exist:", args.load_model)
        print(
            "If you intended to use a local checkpoint, please place the model directory at that path (e.g., unzip t5base_final into the repository)."
        )
        print(
            "Alternatively, provide a Hugging Face model identifier and ensure you have network access and credentials if required."
        )
        sys.exit(1)

# Load the tokenizer and the model that will be tested.
tokenizer = T5Tokenizer.from_pretrained(args.load_model)
print("Loaded tokenizer from directory {}".format(args.load_model))
model = T5ForConditionalGeneration.from_pretrained(args.load_model)
print("Loaded model from directory {}".format(args.load_model))

# Choose device: prefer CUDA if available, otherwise CPU (safer for demo machines)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Hardware Check: Using {DEVICE} for inference.")
model.to(DEVICE)
model.resize_token_embeddings(len(tokenizer))
model.eval()

with open(args.file_path, "r") as f:
    data = json.load(f)
    rule_id = data["rule_id"]
    message = data["message"]
    warning_line = data["warning_line"]
    source_code = data["source_code"]

input_text = (
    "fix "
    + rule_id
    + " "
    + message
    + " "
    + warning_line
    + ":\n"
    + source_code
    + " </s>"
)

predictions = get_single_prediction(
    model,
    tokenizer,
    input_text,
    max_length=256,
    beam_size=args.beam_size,
    num_seq=args.num_seq,
)

print("Input Text:", input_text)

# --- DiffChecker: Output Logic ---
if args.diff or args.save_diff:
    print(f"\n{CYAN}=== DiffChecker: Enhanced Diff Mode Enabled ==={RESET}")
    print(f"Showing top {args.top} predictions...\n")

    diff_buffer = []  # Buffer for file saving

    # Slice predictions based on --top k
    limit = min(args.top, len(predictions))

    for i, pred in enumerate(predictions[:limit]):
        header = f"--- Prediction #{i+1} ---"

        # 1. Console Output (Colored)
        if args.diff:
            diff_text = generate_colored_diff(source_code, pred, use_color=True)
            print(f"{CYAN}{header}{RESET}")
            print(diff_text)
            print()  # Newline padding

        # 2. File Output (Plain text)
        if args.save_diff:
            clean_diff = generate_colored_diff(source_code, pred, use_color=False)
            diff_buffer.append(f"{header}\n{clean_diff}\n")

    # Save to file if requested
    if args.save_diff:
        filename = "pyty_suggestions.diff"
        with open(filename, "w") as f:
            f.write("\n".join(diff_buffer))
        print(f"{GREEN}[+] Diffs saved to {filename}{RESET}")
else:
    # Fallback: simple prediction printout (backward compatibility)
    print("Predictions:")
    for i, pred in enumerate(predictions):
        print(repr(f'      "{i}": "{pred}"'))

# Optionally verify each candidate using static analysis
verification_results = []
if args.verify:
    print("Running static verification on each candidate (this uses pyre or mypy if available)...")
    for i, pred in enumerate(predictions):
        start = time.time()
        try:
            validated, checker_output = verify_candidate(
                source_code, warning_line, pred, rule_id=rule_id, message=message
            )
        except Exception as e:
            validated = False
            checker_output = f"verify_candidate raised exception: {e}"
        elapsed = time.time() - start
        verification_results.append(
            {
                "index": i,
                "prediction": pred,
                "validated": bool(validated),
                "checker_output": checker_output,
                "time_seconds": elapsed,
            }
        )

    # Print validated ones first
    validated_list = [r for r in verification_results if r["validated"]]
    unvalidated_list = [r for r in verification_results if not r["validated"]]

    print("\nValidated predictions:")
    for r in validated_list:
        print(f"  [{r['index']}] validated in {r['time_seconds']:.2f}s: {r['prediction']}")
        # show brief checker output
        print("    checker summary:", (r["checker_output"] or "").splitlines()[:3])

    print("\nUnvalidated predictions:")
    for r in unvalidated_list:
        print(f"  [{r['index']}] NOT validated in {r['time_seconds']:.2f}s: {r['prediction']}")
        print("    checker summary:", (r["checker_output"] or "").splitlines()[:3])

    # Save the verification results to an output file for later inspection
    out_dir = os.path.join(os.getcwd(), "src", "output")
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"predictions_verification_{int(time.time())}.json"
        )
        with open(out_path, "w", encoding="utf-8") as of:
            _json.dump({"input": data, "results": verification_results}, of, indent=2)
        print(f"Verification results saved to {out_path}")
    except Exception as e:
        print("Failed to save verification results:", e)
else:
    # If verification was not requested but we were in diff-only mode,
    # we've already printed some output above. Nothing further to do.
    pass

# If verification was requested but there were no results, still print raw predictions
if args.verify and not verification_results:
    for i, pred in enumerate(predictions):
        print(repr(f'      "{i}": "{pred}"'))
