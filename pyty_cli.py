import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from rich import print, box
from rich.console import Console
from rich.table import Table

console = Console()

def detect_type_error(filepath):
    result = subprocess.run(["mypy", filepath], capture_output=True, text=True)
    output = result.stdout.strip()

    if not output:
        print("[bold green]No type errors found by mypy.[/bold green]")
        return None
    
    lines = output.splitlines()
    first_error = lines[0]
    try:
        location, message = first_error.split(":", maxsplit=2)[1:]
        line_number = int(location.strip())
        return {
            "rule_id": "Mypy Error",
            "message": message.strip(),
            "warning_line": line_number
        }
    except Exception as e:
        print(f"[bold red]Failed to parse mypy output:[/bold red] {first_error}")
        return None

def build_input_json(source_code, error_data):
    return {
        "rule_id": error_data["rule_id"],
        "message": error_data["message"],
        "warning_line": error_data["warning_line"],
        "source_code": source_code
    }

def run_pyty(json_input):
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
        json.dump(json_input, tmp, indent=4)
        tmp.flush()
        tmp_path = tmp.name
    
    print(f"[yellow]Running PyTy on input...[/yellow]")
    subprocess.run(["python3", "pyty_predict.py", "--input", tmp_path])

    Path(tmp_path).unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description="PyTy CLI wrapper for Personal Mode")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to Python file")
    group.add_argument("--code", type=str, help="Inline code snippet")

    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            console.print(f"[bold red]File {args.file} not found.[/bold red]")
            return
        code = path.read_text()
        error_info = detect_type_error(str(path))
    else:
        code = args.code
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp.flush()
            tmp_path = tmp.name
        error_infooooooo = detect_type_error(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
    
    if not error_info:
        print("[red]No usable error information found. Aborting.[/red]")
        return
    
    input_json = build_input_json(code, error_info)
    run_pyty(input_json)

if __name__ == "__main__":
    main()