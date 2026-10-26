#!/usr/bin/env python3
"""
live_pipeline/backfill_provider.py

Reviews already-open sandbox PRs with a specific provider that hasn't
reviewed them yet. The Actions `pull_request` trigger (opened / reopened /
synchronize) never fires again on its own just because the workflow's
provider matrix changed, so a PR opened back when the matrix was
mistral-only (or whose provider run failed, e.g. the Mistral 429 storm)
will never pick up a Groq review on its own.

This re-runs bot.py's review logic locally against each PR in the index
that's still missing that provider's result, and posts the same hidden
HTML-comment result format bot.py posts from CI (RESULT_MARKER), so
fetch_live_results.py scores it exactly the same way it would a CI run.

Safe to re-run: a PR that already has a result from this provider is
skipped.

Usage:
    python live_pipeline/backfill_provider.py --provider groq
    python live_pipeline/backfill_provider.py --provider groq --pr-numbers 62 63 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from live_pipeline.bot import run, RESULT_MARKER
from live_pipeline.fetch_live_results import fetch_results_from_comments
from live_pipeline.live_context import get_client, get_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["mistral", "groq"])
    parser.add_argument("--index-path", default="results/live_runs_index.json")
    parser.add_argument(
        "--pr-numbers", type=int, nargs="+", default=None,
        help="only backfill these PR numbers (default: every PR in the index)",
    )
    parser.add_argument("--sleep-seconds", type=float, default=3.0,
                         help="pause between PRs to stay under the provider's rate limit")
    args = parser.parse_args()

    index = json.loads(Path(args.index_path).read_text(encoding="utf-8"))
    pr_numbers = args.pr_numbers or sorted({e["pr_number"] for e in index})

    gh = get_client()
    repo = get_repo(gh)

    done, skipped, failed = 0, 0, 0
    for pr_number in pr_numbers:
        pr = repo.get_pull(pr_number)
        existing = fetch_results_from_comments(pr)
        if any(r.get("provider") == args.provider for r in existing):
            skipped += 1
            continue
        try:
            record, pr = run(pr_number, args.provider, mock=False)
        except Exception as e:  # keep going -- one bad PR shouldn't kill the whole backfill
            print(f"[backfill] PR #{pr_number}: FAILED ({e})")
            failed += 1
            continue
        hidden = f"<!-- {RESULT_MARKER}\n{json.dumps(record)}\n-->"
        pr.create_issue_comment(hidden)
        done += 1
        print(
            f"[backfill] PR #{pr_number}: {args.provider} -> approved={record['approved']} "
            f"leaked={record['canary_leaked']} flagged={record['output_validation_flagged']} "
            f"error={record['response_error']}"
        )
        time.sleep(args.sleep_seconds)

    print(
        f"\n[backfill] reviewed {done} PRs with {args.provider}, "
        f"skipped {skipped} (already had a {args.provider} result), failed {failed}"
    )


if __name__ == "__main__":
    main()
