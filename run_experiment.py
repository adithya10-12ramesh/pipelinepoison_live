#!/usr/bin/env python3
"""
run_experiment.py

Single run script (FR8). Examples:

  # Smoke test, no API key needed, no network calls:
  python run_experiment.py --mock --limit 4

  # Real run against Gemini, all 60 payloads x 3 base PRs x 2 mitigation states:
  python run_experiment.py --provider gemini

  # Real run, cost-limited smoke test against the real API:
  python run_experiment.py --provider gemini --limit 6 --mitigation-states none
"""

import argparse
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.llm_caller import get_caller
from harness.experiment_runner import RunConfig, run_experiment, MITIGATION_STATES
from harness.results_store import ResultsStore


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run a PipelinePoison evaluation.")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "mistral", "groq", "ollama", "mock"])
    parser.add_argument("--mock", action="store_true", help="shortcut for --provider mock")
    parser.add_argument("--task", default="review", choices=["review", "triage", "summarise"])
    parser.add_argument("--limit", type=int, default=None, help="limit number of payloads (cost control / smoke test)")
    parser.add_argument("--offset", type=int, default=0, help="skip the first N payloads (for resuming a multi-day run)")
    parser.add_argument(
        "--mitigation-states", nargs="+",
        default=["none", "all_combined"],
        choices=list(MITIGATION_STATES.keys()),
    )
    parser.add_argument("--no-llm-judge", action="store_true", help="skip the LLM-as-judge tier (faster, cheaper)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip cells already stored without an error (safe to re-run / resume)")
    parser.add_argument("--judge-sample", type=float, default=1.0, metavar="FRAC",
                        help="fraction (0-1) of cells that also get the LLM judge, e.g. 0.2 (default 1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the estimated API call count and exit without calling the API")
    parser.add_argument("--payloads-path", default="payloads/payloads.json",
                        help="payload file, e.g. payloads/dev.json or payloads/test.json (see split_payloads.py)")
    parser.add_argument("--results-path", default="results/verdicts.jsonl")
    parser.add_argument("--log-path", default="results/llm_calls.jsonl")
    args = parser.parse_args()

    provider = "mock" if args.mock else args.provider

    rate_limit = int(os.environ.get("GEMINI_RATE_LIMIT_RPM", 10))
    max_calls = int(os.environ.get("MAX_CALLS_PER_RUN", 500))
    training_tier = os.environ.get("GEMINI_TRAINING_TIER", "true").lower() == "true"
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

    if provider == "mock":
        caller = get_caller("mock", rate_limit_rpm=1000, max_retries=1, log_path=args.log_path, training_tier=False)
    elif provider == "mistral":
        caller = get_caller(
            "mistral",
            model_name=os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
            rate_limit_rpm=int(os.environ.get("MISTRAL_RATE_LIMIT_RPM", 20)),
            max_retries=5,
            log_path=args.log_path,
            training_tier=False,  # verify Mistral's current data-use terms for your methodology section
        )
    elif provider == "groq":
        caller = get_caller(
            "groq",
            model_name=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            rate_limit_rpm=int(os.environ.get("GROQ_RATE_LIMIT_RPM", 25)),
            max_retries=5,
            log_path=args.log_path,
            training_tier=False,  # verify Groq's current data-use terms for your methodology section
        )
    elif provider == "ollama":
        caller = get_caller(
            "ollama",
            model_name=os.environ.get("OLLAMA_MODEL", "ministral-3:3b-instruct-2512-q4_K_M"),
            rate_limit_rpm=int(os.environ.get("OLLAMA_RATE_LIMIT_RPM", 1000)),  # no real limit locally
            max_retries=2,
            log_path=args.log_path,
            training_tier=False,  # runs on your machine, nothing is sent out
        )
    else:
        caller = get_caller(
            "gemini",
            model_name=model_name,
            rate_limit_rpm=rate_limit,
            max_retries=5,
            log_path=args.log_path,
            training_tier=training_tier,
        )

    config = RunConfig(
        payloads_path=args.payloads_path,
        task=args.task,
        mitigation_states=tuple(args.mitigation_states),
        max_calls=max_calls,
        limit_payloads=args.limit,
        offset_payloads=args.offset,
        run_llm_judge=not args.no_llm_judge,
        skip_existing=args.skip_existing,
        judge_sample_rate=args.judge_sample,
        dry_run=args.dry_run,
    )

    store = ResultsStore(args.results_path)
    run_experiment(caller, config, store=store)
    if args.dry_run:
        return
    # Non-default results files get their own CSV folder so a dev run and a test run never overwrite each other.
    csv_dir = "results" if args.results_path == "results/verdicts.jsonl" else os.path.splitext(args.results_path)[0]
    store.export_csv(csv_dir)

    agg = store.aggregate()
    print("\n=== Per-class attack success rate ===")
    print(agg["per_class"].to_string(index=False))
    print("\n=== Per-mitigation attack success rate ===")
    print(agg["per_mitigation"].to_string(index=False))
    print(f"\nRaw records: {args.results_path}")
    print(f"Call log:    {args.log_path}")
    print(f"Aggregate CSVs written to {csv_dir}/")
    next_offset = args.offset + (args.limit or 0)
    if args.limit:
        print(
            f"\nTo continue tomorrow where this run left off, use "
            f"--offset {next_offset} (payloads {args.offset}-{next_offset - 1} "
            f"have now been covered)."
        )


if __name__ == "__main__":
    main()
