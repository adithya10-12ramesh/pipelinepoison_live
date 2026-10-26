#!/usr/bin/env python3
"""
live_pipeline/fetch_live_results.py

Fetch bot results from PR comments, score them with canary_judge.py, and
append them to results_live/verdicts.jsonl.

By default, both Groq and Mistral results are processed. Use --provider groq
or --provider mistral to process only one provider. Re-runs are safe because
results are deduplicated by (PR number, provider).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from harness.results_store import ResultsStore
from live_pipeline.live_context import get_client, get_repo
from live_pipeline.canary_judge import score
from live_pipeline.bot import RESULT_MARKER

RESULT_RE = re.compile(rf"<!-- {re.escape(RESULT_MARKER)}\n(.*?)\n-->", re.DOTALL)


def fetch_results_from_comments(pr) -> list[dict]:
    results = []
    for comment in pr.get_issue_comments():
        match = RESULT_RE.search(comment.body or "")
        if match:
            try:
                results.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                continue
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch and score PR review bot results from GitHub comments."
    )
    parser.add_argument(
        "--provider",
        choices=("groq", "mistral", "all"),
        default="all",
        help="Only process this provider, or 'all' for both (default: all).",
    )
    parser.add_argument(
        "--index",
        default=os.environ.get("LIVE_RUNS_INDEX", "results/live_runs_index.json"),
        help="Path to the live PR runs index (default: LIVE_RUNS_INDEX or results/live_runs_index.json).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    index_path = args.index
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))

    gh = get_client()
    repo = get_repo(gh)
    store = ResultsStore("results_live/verdicts.jsonl")

    # Deduplicate previously scored (PR number, provider) pairs.
    existing_df = store.load()
    already_scored = set()
    if (
        not existing_df.empty
        and "pr_number" in existing_df.columns
        and "provider" in existing_df.columns
    ):
        already_scored = set(
            zip(existing_df["pr_number"].tolist(), existing_df["provider"].tolist())
        )

    allowed_providers = {"groq", "mistral"} if args.provider == "all" else {args.provider}
    prs_with_result = 0
    pending = 0
    newly_scored = 0
    already_had = 0

    for entry in index:
        pr = repo.get_pull(entry["pr_number"])
        all_results = fetch_results_from_comments(pr)
        results = [
            result
            for result in all_results
            if str(result.get("provider", "")).lower() in allowed_providers
        ]

        if not results:
            pending += 1
            continue

        prs_with_result += 1
        for bot_result in results:
            provider = str(bot_result.get("provider", "")).lower()
            key = (pr.number, provider)
            if key in already_scored:
                already_had += 1
                continue

            store.append(score(bot_result, entry))
            already_scored.add(key)
            newly_scored += 1

        found = {str(result.get("provider", "")).lower() for result in results}
        missing = allowed_providers - found
        if missing:
            print(
                f"[fetch_live_results] PR #{pr.number}: have {found}, "
                f"still waiting on {missing}"
            )

    provider_label = args.provider
    print(
        f"\n[fetch_live_results] {prs_with_result} PRs have at least one "
        f"{provider_label} result ({newly_scored} newly scored this run, "
        f"{already_had} already scored on a previous run), "
        f"{pending} have no matching {provider_label} result yet"
    )
    store.export_csv(out_dir="results_live")


if __name__ == "__main__":
    main()
