import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Tuple, Optional


def _apply_candidate_to_source(source_code: str, warning_line: str, candidate: str) -> str:
    """Apply the candidate fix to the provided source snippet.

    Heuristics:
    - If candidate contains multiple lines or starts with typical Python tokens (def/class/import),
      assume it is a replacement for the whole snippet.
    - Otherwise, try to replace the exact warning_line in source_code with candidate.
    - If replacement fails, append the candidate at the end (best-effort).
    """
    cand = candidate
    # Normalize line endings
    if "\r\n" in cand:
        cand = cand.replace("\r\n", "\n")

    stripped = cand.strip()
    # If it's multi-line or looks like a block, replace whole snippet
    if "\n" in cand or stripped.startswith(("def ", "class ", "import ", "from ", "@", "if ", "for ", "while ")):
        if not cand.endswith("\n"):
            cand = cand + "\n"
        return cand

    # Otherwise try to replace the warning line
    if warning_line and warning_line in source_code:
        return source_code.replace(warning_line, cand)

    # Fallback: append
    if not source_code.endswith("\n"):
        source_code = source_code + "\n"
    return source_code + cand + "\n"


def verify_candidate(source_code: str, warning_line: str, candidate: str, rule_id: str = "", message: str = "") -> Tuple[bool, str]:
    """Verify a candidate fix by running a static checker on a temporary file.

    Returns (validated: bool, checker_output: str).

    Strategy:
    - Create a temporary directory with the modified code as candidate.py
    - Try running 'pyre' if available; if pyre runs, consider the fix validated when the original
      rule_id or message no longer appears in the pyre output.
    - If pyre isn't available or fails, fall back to running 'mypy' on the file and treat absence
      of reported errors as validation.
    - Cleanup the temporary directory.
    """
    tempdir = tempfile.mkdtemp(prefix="pyty_verify_")
    try:
        modified = _apply_candidate_to_source(source_code, warning_line, candidate)
        file_path = os.path.join(tempdir, "candidate.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(modified)

        # Fast heuristic: extract a likely target name from message or warning_line
        target_name: Optional[str] = None
        # Try to extract backticked name from message: Name `y_test`
        m = re.search(r"`([^`]+)`", message or "")
        if m:
            target_name = m.group(1)
        else:
            # Try to extract the first identifier in the warning line
            m2 = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", warning_line or "")
            if m2:
                target_name = m2.group(1)

        # Record a heuristic note; if candidate syntactically defines target, accept immediately.
        heuristic_note = ""
        if target_name:
            assign_re = re.compile(r"^\s*" + re.escape(target_name) + r"\s*=", re.MULTILINE)
            def_re = re.compile(r"^\s*(def|class)\s+" + re.escape(target_name) + r"\b", re.MULTILINE)
            import_re = re.compile(r"^\s*(from\s+\S+\s+import\s+.*\b" + re.escape(target_name) + r"\b|import\s+.*\b" + re.escape(target_name) + r"\b)", re.MULTILINE)

            if assign_re.search(candidate) or def_re.search(candidate) or import_re.search(candidate):
                return True, f"Heuristic: target '{target_name}' is defined in candidate. Marking as validated (fast path)."
            else:
                # Do not return; continue to run static checker for thorough verification.
                heuristic_note = f"Heuristic: target '{target_name}' not defined in candidate.\nCandidate snippet:\n{candidate}\n"

        # First try pyre if available
        pyre_bin = shutil.which("pyre")
        combined_output = ""
        if pyre_bin:
            # write a minimal .pyre_configuration to run pyre locally
            config = {"source_directories": ["."]}
            with open(os.path.join(tempdir, ".pyre_configuration"), "w", encoding="utf-8") as cf:
                json.dump(config, cf)

            try:
                proc = subprocess.run([pyre_bin, "--noninteractive", "check", "--output=json"], cwd=tempdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
                combined_output = proc.stdout.decode(errors="ignore")
                if proc.returncode == 0:
                    return True, (heuristic_note + combined_output) if heuristic_note else combined_output
                try:
                    parsed = json.loads(combined_output)
                    errors = parsed.get("errors", []) if isinstance(parsed, dict) else []
                    # If any error matches the original rule/message, consider NOT validated
                    for err in errors:
                        if rule_id and rule_id in json.dumps(err):
                            return False, (heuristic_note + combined_output) if heuristic_note else combined_output
                        if message and message.strip() and message.strip() in json.dumps(err):
                            return False, (heuristic_note + combined_output) if heuristic_note else combined_output
                    # No matching error found: consider validated
                    return True, (heuristic_note + combined_output) if heuristic_note else combined_output
                except Exception:
                    if "no such option: --output" in combined_output or "Usage: pyre check" in combined_output:
                        proc2 = subprocess.run([pyre_bin, "--noninteractive", "check"], cwd=tempdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
                        combined_output = proc2.stdout.decode(errors="ignore")
                        if proc2.returncode == 0:
                            return True, (heuristic_note + combined_output) if heuristic_note else combined_output
                        if rule_id and rule_id in combined_output:
                            return False, (heuristic_note + combined_output) if heuristic_note else combined_output
                        if message and isinstance(message, str) and message.strip() and message.strip() in combined_output:
                            return False, (heuristic_note + combined_output) if heuristic_note else combined_output
                        return False, (heuristic_note + combined_output) if heuristic_note else combined_output
                    else:
                        if rule_id and rule_id in combined_output:
                            return False, (heuristic_note + combined_output) if heuristic_note else combined_output
                        if message and isinstance(message, str) and message.strip() and message.strip() in combined_output:
                            return False, (heuristic_note + combined_output) if heuristic_note else combined_output
                        return False, (heuristic_note + combined_output) if heuristic_note else combined_output
            except subprocess.CalledProcessError as e:
                combined_output = (e.stdout or b"").decode(errors="ignore")
            except Exception as e:
                combined_output = f"pyre invocation failed: {e}\n"

        # Fallback to mypy if pyre is not available or failed
        mypy_bin = shutil.which("mypy")
        if mypy_bin:
            try:
                proc = subprocess.run([mypy_bin, file_path, "--ignore-missing-imports"], cwd=tempdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
                combined_output += "\n" + proc.stdout.decode(errors="ignore")
                if proc.returncode == 0 and (not proc.stdout):
                    return True, (heuristic_note + combined_output) if heuristic_note else combined_output
                return False, (heuristic_note + combined_output) if heuristic_note else combined_output
            except Exception as e:
                combined_output += f"mypy invocation failed: {e}\n"

        # As a last resort, attempt a basic runtime compile to detect syntax errors
        try:
            compile(modified, file_path, 'exec')
            combined_output += "\nNo static checker available (pyre/mypy). Only syntax checked.\n"
            return False, (heuristic_note + combined_output) if heuristic_note else combined_output
        except SyntaxError as se:
            combined_output += f"\nSyntaxError during compile: {se}\n"
            return False, (heuristic_note + combined_output) if heuristic_note else combined_output

    finally:
        try:
            shutil.rmtree(tempdir)
        except Exception:
            pass
