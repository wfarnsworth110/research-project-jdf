#!/usr/bin/env python3
"""
Post-process existing error_removal_analysis JSON (or test_data) to run static verification
on each pred_top-* entry using verify_fix.verify_candidate and augment the JSON with
checker output and updated warning_removed flags.

Usage:
  python src/verify_predictions_batch.py --input src/output/test_data_baseline_top50.json \
      --out src/output/error_removal_analysis_baseline_top50_verified.json --limit 100

If you pass an existing error_removal_analysis file as input it will update its entries in-place
and write to the output path. If you pass a test_data file (with 'predictions' or 'top_five'), the
script will attempt to construct a compatible output.

This script is intentionally conservative and can be slow if run over many files (runs static
checker per candidate). Use --parallel to speed up verification.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from verify_fix import verify_candidate


def verify_one(pred_parsed, source_code, warning_line, rule_id, message):
    start = time.time()
    validated, checker_output = verify_candidate(source_code, warning_line, pred_parsed, rule_id=rule_id, message=message)
    return {
        'validated': bool(validated),
        'checker_output': checker_output,
        'time_seconds': time.time() - start,
    }


def process_error_removal(input_path, output_path, limit=None, parallel=1):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    if limit:
        total = min(total, limit)

    print(f"Processing {total} entries from {input_path} (limit={limit})")

    start_all = time.time()
    results = []

    # Prepare jobs
    jobs = []
    for idx, entry in enumerate(data[:total]):
        rule = entry.get('warning_type') or entry.get('linter_report', {}).get('rule_id','')
        message = entry.get('warning_message') or entry.get('linter_report', {}).get('message','')
        warning_line = entry.get('warning_line') or entry.get('linter_report',{}).get('line_begin','')
        source_code = entry.get('source_code') or entry.get('source_code','')

        # collect pred_top-* keys
        preds = []
        for k in range(1,51):
            key = f'pred_top-{k}'
            if key in entry:
                pred_parsed = entry[key].get('pred_parsed') or entry[key].get('pred')
                preds.append((key, pred_parsed))

        jobs.append((idx, entry, rule, message, warning_line, source_code, preds))

    # Worker
    def worker(job):
        idx, entry, rule, message, warning_line, source_code, preds = job
        for key, pred_parsed in preds:
            try:
                v = verify_one(pred_parsed, source_code, warning_line, rule, message)
                # Update entry structure similar to existing error_removal_analysis format
                entry.setdefault(key, {})
                entry[key]['warning_removed'] = bool(v['validated'])
                entry[key]['checker_output'] = v['checker_output']
                entry[key]['verification_time'] = v['time_seconds']
                entry[key]['fail_reason'] = 'N/A' if v['validated'] else 'Error Still Exists or New Errors'
            except Exception as e:
                entry.setdefault(key, {})
                entry[key]['warning_removed'] = False
                entry[key]['checker_output'] = f'verify_exception: {e}'
                entry[key]['verification_time'] = 0.0
                entry[key]['fail_reason'] = 'verify_exception'
        return entry

    if parallel and parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as exe:
            future_to_job = {exe.submit(worker, job): job for job in jobs}
            for i, fut in enumerate(as_completed(future_to_job)):
                job = future_to_job[fut]
                try:
                    updated = fut.result()
                except Exception as e:
                    print('Job failed:', e)
                if (i+1) % 10 == 0:
                    print(f'Completed {i+1}/{len(jobs)} jobs')
    else:
        for i, job in enumerate(jobs):
            worker(job)
            if (i+1) % 10 == 0:
                print(f'Completed {i+1}/{len(jobs)} jobs')

    # Write output
    with open(output_path, 'w', encoding='utf-8') as of:
        json.dump(data, of, indent=2)

    print(f'Wrote verified output to {output_path} in {time.time()-start_all:.1f}s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--out', '-o', required=True)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of entries to process (for quick demos)')
    parser.add_argument('--parallel', type=int, default=1, help='Number of threads for verification')
    args = parser.parse_args()

    process_error_removal(args.input, args.out, limit=args.limit, parallel=args.parallel)


if __name__ == '__main__':
    main()
