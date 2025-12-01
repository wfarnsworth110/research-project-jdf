import argparse
import json
import sys
import difflib  # DiffChecker dependency

sys.path.append("..")

from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import set_seed
import torch

from utils import boolean_string
from utils import get_current_time

# --- DiffChecker: Color Constants ---
GREEN = '\033[32m'
RED = '\033[91m'
RESET = '\033[0m'
CYAN = '\033[96m'

def generate_colored_diff(original: str, fixed: str, use_color=True) -> str:
    """
    Generates a unified diff between original code and the predicted fix.
    """
    original_lines = original.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)
    
    # Generate unified diff with 3 lines of context
    diff = difflib.unified_diff(
        original_lines, 
        fixed_lines, 
        fromfile='Original', 
        tofile='Prediction', 
        n=3 
    )
    
    output = []
    for line in diff:
        if use_color:
            if line.startswith('+') and not line.startswith('+++'):
                output.append(GREEN + line.strip() + RESET)
            elif line.startswith('-') and not line.startswith('---'):
                output.append(RED + line.strip() + RESET)
            elif line.startswith('^'):
                output.append(line.strip())
            else:
                output.append(line.strip())
        else:
            output.append(line.strip())
            
    return "\n".join(output)

def get_single_prediction(model, tokenizer, input_text, max_length=256, beam_size=50, num_seq=50):
    # Tokenize the input text
    # Note: .to(model.device) ensures inputs go to the same hardware (CPU or GPU) as the model
    input_ids = tokenizer.encode(input_text, truncation=True, padding=True, return_tensors='pt').to(model.device)
    
    # Generate predictions
    beam_outputs = model.generate(
        input_ids, 
        max_length=max_length, 
        num_beams=beam_size, 
        num_return_sequences=num_seq,
        early_stopping=False
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
    required=True,
)
parser.add_argument(
    "-lm", "--load-model", type=str, default=""
) 
parser.add_argument(
    "-ea", "--eval-all", type=boolean_string, default=False
)
parser.add_argument("-eas", "--eval-acc-steps", type=int, default=1)
parser.add_argument("-et", "--error-type", type=str, default="")
parser.add_argument("-bm", "--beam-size", type=int, default=50) 
parser.add_argument("-seq", "--num-seq", type=int, default=50) 
parser.add_argument("-f", "--file_path", type=str, required=True, help="Enter the path to the file containing input.")

# --- DiffChecker: New Arguments ---
parser.add_argument("--diff", action="store_true", help="Enable enhanced diff output (DiffChecker)")
parser.add_argument("--save-diff", action="store_true", help="Save the diffs to a file named pyty_suggestions.diff")
parser.add_argument("--top", type=int, default=50, help="Limit the number of suggestions to display/save")

args = parser.parse_args()

model_name = args.model_name

# Load the tokenizer and the model that will be tested.
tokenizer = T5Tokenizer.from_pretrained(args.load_model)
print("Loaded tokenizer from directory {}".format(args.load_model))
model = T5ForConditionalGeneration.from_pretrained(args.load_model)
print("Loaded model from directory {}".format(args.load_model))

# --- BUG FIX: INTELLIGENT DEVICE SELECTION ---
# Original code forced CUDA, which crashes on laptops without NVIDIA GPUs.
# This block checks for a GPU first, then falls back to CPU if needed.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Check: Using {device} for inference.")
model.to(device)
# ---------------------------------------------

model.resize_token_embeddings(len(tokenizer))
model.eval()

with open(args.file_path, 'r') as f:
    data = json.load(f)
    rule_id = data['rule_id']
    message = data['message']
    warning_line = data['warning_line']
    source_code = data['source_code']

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

predictions = get_single_prediction(model, tokenizer, input_text, max_length=256, beam_size=50, num_seq=50)

# --- DiffChecker: Output Logic ---
print("Input Text:", input_text)

if args.diff or args.save_diff:
    print(f"\n{CYAN}=== DiffChecker: Enhanced Diff Mode Enabled ==={RESET}")
    print(f"Showing top {args.top} predictions...\n")
    
    diff_buffer = [] # Buffer for file saving
    
    # Slice predictions based on --top k
    limit = min(args.top, len(predictions))
    
    for i, pred in enumerate(predictions[:limit]):
        # Header
        header = f"--- Prediction #{i+1} ---"
        
        # 1. Console Output (Colored)
        if args.diff:
            diff_text = generate_colored_diff(source_code, pred, use_color=True)
            print(f"{CYAN}{header}{RESET}")
            print(diff_text)
            print() # Newline padding
            
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
    # --- Original Behavior (Backward Compatibility) ---
    print("Predictions:")
    for i, pred in enumerate(predictions):
        print(repr(f"      \"{i}\": \"{pred}\""))