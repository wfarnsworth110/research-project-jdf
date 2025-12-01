Static verification of PyTy fixes (FAST MODE)

This file documents how to run the new static verification feature that checks whether a candidate fix actually resolves the original Pyre-reported type error.

Overview
- For each top-K candidate produced by `pyty_predict.py`, the verification step writes a temporary file with the candidate applied and runs `pyre` (preferred) or `mypy` (fallback).
- The script marks a candidate as validated when the static checker does not report the original error (or when a syntactic heuristic is satisfied for common cases like "Unbound name").

Quick demo (Docker)

1) Build the Docker image (from the repository root):

```bash
docker build -t icse2024 .
```

2) Run `pyty_predict.py` with verification enabled (default). This command runs inside the container and mounts the repo so output files are saved to the host copy.

```bash
docker run --rm -v $(pwd):/workspace -w /workspace icse2024 python src/pyty_predict.py -mn t5base_final -lm t5base_final/checkpoint-1190 -f src/predict_sample_input.json -vf True
```

Notes for recording the demo:
- The script prints validated and unvalidated predictions separately.
- A JSON file with detailed verification results is saved to `src/output/predictions_verification_<timestamp>.json`.
- If you prefer to skip verification for a faster run, pass `-vf False`.

If you want me to build and run the Docker image in this environment now, tell me and I'll trigger the build and a quick prediction run. Building the full image may take a long time because it installs large ML dependencies (PyTorch, Transformers). If you want a fast local trial without Docker, I can also run a small Python test that demonstrates the verification logic on a synthetic example.
