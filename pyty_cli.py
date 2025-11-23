import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
PYTY_SCRIPT = SRC_DIR / "pyty_predict.py"

# Personal mode info from README
DEFAULT_MODEL_NAME = "t5base_final"
DEFAULT_MODEL_PATH = "t5base_final/checkpoint-1190"

MYPY_ERROR_RE = re.compile(
    r"^(?P<file>.*?):(?P<line>\d+):(?P<col>\d+):\s*error:\s*(?P<message>.*?)(?:\s+\[(?P<code>[^\]]+)\])?$"
)


def run_mypy_on_file(path: Path) -> str:
    """Run mypy on the given file and return stdout as a string."""
    try:
        result = subprocess.run(
            ["mypy", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("Error: `mypy` is not installed or not on PATH.", file=sys.stderr)
        sys.exit(1)

    output = (result.stdout or "") + (result.stderr or "")
    return output


def extract_error_info(
    mypy_output: str, source_path: Path, explicit_source: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Parse mypy output and extract the first type error as a dict suitable for
    PyTy PERSONAL MODE input.

    Returns:
        dict with keys: rule_id, message, warning_line, source_code or None
        if there are no issues (Success: no issues found ...).

    Raises:
        ValueError: if output doesn't contain success line or parseable error.
    """
    lines = mypy_output.splitlines()

    # Try to find first actual error line
    for line in lines:
        match = MYPY_ERROR_RE.match(line.strip())
        if not match:
            continue

        file_path = match.group("file")
        line_no = int(match.group("line"))
        message = match.group("message").strip()
        code = match.group("code")

        # Use mypy's code (e.g. [name-defined]) as "rule_id" if present,
        # otherwise fall back to a generic one
        rule_id = code if code else "mypy-error"

        # Load source text
        if explicit_source is not None and Path(file_path) == source_path:
            # Create temp file from --code snippet
            source_text = explicit_source
            source_lines = source_text.splitlines(keepends=True)
        else:
            try:
                with open(source_path, "r", encoding="utf-8") as f:
                    source_text = f.read()
                source_lines = source_text.splitlines(keepends=True)
            except OSError:
                source_text = ""
                source_lines = []

        if 1 <= line_no <= len(source_lines):
            warning_line = source_lines[line_no - 1].rstrip("\n")
        else:
            warning_line = ""

        return {
            "rule_id": rule_id,
            "message": message,
            "warning_line": warning_line,
            "source_code": source_text,
        }

    # No parseable error -> either clean code or some unexpected output
    if "Success: no issues found" in mypy_output:
        return None

    # If we get here, the output was not a clean run or a parseable error line
    raise ValueError(mypy_output.strip() or "No usable mypy output.")


def write_pyty_input_json(error_info: Dict[str, Any]) -> Path:
    """PyTy input JSON -> temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    with tmp:
        json.dump(
            {
                "rule_id": error_info["rule_id"],
                "message": error_info["message"],
                "warning_line": error_info["warning_line"],
                "source_code": error_info["source_code"],
            },
            tmp,
            indent=2,
        )
    return Path(tmp.name)


def run_pyty(json_path: Path, model_name: str, model_path: str) -> int:
    """
    Invoke PyTy's PERSONAL MODE prediction script in this repo.

    We run:
        python pyty_predict.py -mn <model_name> -lm <model_path> -f <json_path>
    with cwd=src/ so that model paths like t5base_final/checkpoint-1190 resolve.
    """
    if not PYTY_SCRIPT.is_file():
        print(
            f"Error: cannot find pyty_predict.py at {PYTY_SCRIPT}. "
            "Make sure you're running this from the repository root.",
            file=sys.stderr,
        )
        return 1

    cmd = [
        sys.executable,
        str(PYTY_SCRIPT),
        "-mn",
        model_name,
        "-lm",
        model_path,
        "-f",
        str(json_path),
    ]

    print("Running PyTy on input...")
    result = subprocess.run(
        cmd,
        cwd=str(SRC_DIR),  # match original instructions (run from ./src/)
        text=True,
        capture_output=True,
    )

    # Stream PyTy's output for visibility
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"PyTy exited with code {result.returncode}", file=sys.stderr)

    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive CLI wrapper for PyTy PERSONAL MODE.\n"
            "Given a Python file or snippet, this tool:\n"
            "   1) runs mypy to find a type error,\n"
            "   2) builds the JSON required by PyTy, and\n"
            "   3) invokes PyTy's pyty_predict.py with the trained model."
        )
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to a Python file to analyze.",
    )
    group.add_argument(
        "--code",
        "-c",
        type=str,
        help="Inline Python snippet to analyze (as a string).",
    )

    parser.add_argument(
        "--model-name",
        "-mn",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name for PyTy (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--model-path",
        "-lm",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Model checkpoint path relative to src/ "
            f"(default: {DEFAULT_MODEL_PATH})."
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete temporary files (for debugging).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    temp_source_path: Optional[Path] = None
    explicit_source: Optional[str] = None

    # 1. Obtain a concrete file path and source text
    if args.file:
        source_path = (REPO_ROOT / args.file).resolve()
        if not source_path.is_file():
            print(f"Error: file not found: {source_path}", file=sys.stderr)
            sys.exit(1)
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                explicit_source = f.read()
        except OSError as e:
            print(f"Error reading file {source_path}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # --code: write the snippet to a temporary .py file
        explicit_source = args.code or ""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        with tmp:
            tmp.write(explicit_source)
        temp_source_path = Path(tmp.name)
        source_path = temp_source_path

    # 2. Run mypy
    mypy_output = run_mypy_on_file(source_path)

    # 3. Extract error information
    try:
        error_info = extract_error_info(
            mypy_output, source_path=source_path, explicit_source=explicit_source
        )
    except ValueError as e:
        print(f"Failed to parse mypy output: {e}", file=sys.stderr)
        print("Raw mypy output:")
        print(mypy_output.strip())
        if temp_source_path and not args.keep_temp:
            temp_source_path.unlink(missing_ok=True)
        sys.exit(1)

    if error_info is None:
        # Clean code, nothing to send to PyTy
        print("mypy reports no type errors. Nothing to repair.")
        if temp_source_path and not args.keep_temp:
            temp_source_path.unlink(missing_ok=True)
        sys.exit(0)

    # 4. Write PyTy input JSON
    json_path = write_pyty_input_json(error_info)
    print(f"Generated PyTy input JSON at: {json_path}")

    # 5. Run PyTy
    exit_code = run_pyty(
        json_path=json_path,
        model_name=args.model_name,
        model_path=args.model_path,
    )

    # 6. Cleanup temps if requested
    if not args.keep_temp:
        json_path.unlink(missing_ok=True)
        if temp_source_path:
            temp_source_path.unlink(missing_ok=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
