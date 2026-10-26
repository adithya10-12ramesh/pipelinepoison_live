#!/usr/bin/env python3
"""
live_pipeline/open_prs.py

Opens real pull requests against the sandbox repo so the live bot
(.github/workflows/pr_review_bot.yml -> bot.py) reviews real GitHub
content instead of a simulated prompt. Injects a payload into exactly
one surface per PR (PR_DESCRIPTION / COMMIT_MESSAGE / SOURCE_DIFF /
CI_LOG_OUTPUT), matching harness/taxonomy.py's Surface enum, so live
results line up with the same 6x4 taxonomy as the simulated runs.

Real PRs cost GitHub Actions minutes and API calls, so this does NOT try
to cover all 60 payloads x all base PRs like the offline harness -- it
runs a stratified sample (by default: 1 payload per attack class x all
5 mitigation states = 120 PRs, x2 providers reviewing each = 240 bot
runs). Use --per-class / --mitigation-states to shrink further for a
first smoke test; start with --per-class 1 --mitigation-states none.

SAFETY: only ever point this at a repo you own (see paper/ethics_note.md).
LIVE_GITHUB_TOKEN needs `repo` scope on that repo and nothing else.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from github import GithubException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from harness.experiment_runner import MITIGATION_STATES, load_payloads
from harness.taxonomy import ALL_CLASSES

from live_pipeline.live_context import get_client, get_repo, CI_LOG_PATH, ensure_label

PAYLOAD_TARGET_PATH = "sandbox/payload_target.py"
DEFAULT_BASE_BRANCH = "main"


def load_base_prs(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # fixtures/base_prs.json (the offline harness's fixture) has no
    # commit_message or true_severity field -- backfill sensible defaults
    # rather than requiring you to hand-edit that file for live use too.
    for bp in data:
        bp.setdefault("true_severity", "medium")
        bp.setdefault("commit_message", bp.get("title", "Update"))
        bp.setdefault("diff_marker_content", f"# {bp.get('id', 'baseline')}\n")
    return data


def stratified_sample(payloads: list[dict], per_class: int, seed: int = 0, classes: list[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for p in payloads:
        by_class.setdefault(p["class_id"], []).append(p)
    sample = []
    for class_id in (classes or ALL_CLASSES):
        candidates = by_class.get(class_id, [])
        if candidates:
            sample.extend(rng.sample(candidates, min(per_class, len(candidates))))
    return sample


def inject(payload: dict, base_pr: dict, mitigation_state: str, run_index: int, canary_value: str):
    surface = payload["surface"]
    branch = f"ppz/{payload['class_id'].lower()}-{mitigation_state}-{run_index}"

    title = base_pr.get("title", "Update")
    body = base_pr.get("body", "")
    commit_message = base_pr["commit_message"]
    ci_log_content = base_pr.get("ci_log", "Build succeeded.\n")
    diff_marker = base_pr["diff_marker_content"]

    if surface == "PR_DESCRIPTION":
        body = f"{body}\n\n{payload['text']}"
    elif surface == "COMMIT_MESSAGE":
        commit_message = f"{commit_message}\n\n{payload['text']}"
    elif surface == "SOURCE_DIFF":
        diff_marker = f"{diff_marker}\n# {payload['text']}\n"
    elif surface == "CI_LOG_OUTPUT":
        ci_log_content = f"{ci_log_content}\n{payload['text']}\n"

    # Every PR gets a canary line in its CI log (not just EXFIL_SECRET
    # cells), so the bot always has something real to leak if coerced.
    ci_log_content += f"\nDEPLOY_TOKEN={canary_value}\n"

    file_changes = {PAYLOAD_TARGET_PATH: diff_marker, CI_LOG_PATH: ci_log_content}
    return branch, title, body, commit_message, file_changes


def commit_files(repo, branch: str, commit_message: str, file_changes: dict[str, str]):
    base = repo.get_branch(DEFAULT_BASE_BRANCH)
    try:
        repo.create_git_ref(ref=f"refs/heads/{branch}", sha=base.commit.sha)
    except GithubException as e:
        if e.status == 422 and "already exists" in str(e.data.get("message", "")):
            repo.get_git_ref(f"heads/{branch}").edit(sha=base.commit.sha, force=True)
        else:
            raise
    for path, content in file_changes.items():
        try:
            existing = repo.get_contents(path, ref=branch)
            repo.update_file(path, commit_message, content, existing.sha, branch=branch)
        except Exception:
            repo.create_file(path, commit_message, content, branch=branch)


def get_existing_open_pr(repo, branch: str):
    """If a PR for this branch is already open (e.g. a previous run got
    interrupted after creating it but before the index was written),
    return it instead of trying -- and failing -- to create a duplicate."""
    owner = repo.owner.login
    open_prs = repo.get_pulls(state="open", head=f"{owner}:{branch}")
    for pr in open_prs:
        return pr
    return None


def create_pull_idempotent(repo, branch: str, title: str, body: str):
    try:
        return repo.create_pull(title=title, body=body, head=branch, base=DEFAULT_BASE_BRANCH)
    except GithubException as e:
        msg = str(e.data.get("errors", e.data.get("message", "")))
        if e.status == 422 and "already exists" in msg:
            existing = get_existing_open_pr(repo, branch)
            if existing is not None:
                return existing
        raise


def open_one_pr(repo, payload: dict, base_pr: dict, mitigation_state: str, run_index: int, canary_value: str):
    branch, title, body, commit_message, file_changes = inject(payload, base_pr, mitigation_state, run_index, canary_value)
    commit_files(repo, branch, commit_message, file_changes)
    pr = create_pull_idempotent(repo, branch, title, body)
    label = f"mitigation:{mitigation_state}"
    ensure_label(repo, label)
    ensure_label(repo, "live-eval")
    pr.add_to_labels(label, "live-eval")
    return pr


def open_one_benign_pr(repo, benign: dict, mitigation_state: str, run_index: int):
    branch = f"ppz/benign-{run_index}"
    commit_files(repo, branch, "benign update", {PAYLOAD_TARGET_PATH: benign.get("diff_marker_content", "# benign\n")})
    pr = create_pull_idempotent(repo, branch, benign["title"], benign.get("body", ""))
    label = f"mitigation:{mitigation_state}"
    ensure_label(repo, label)
    ensure_label(repo, "live-eval")
    ensure_label(repo, "benign")
    pr.add_to_labels(label, "live-eval", "benign")
    return pr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads-path", default="payloads/payloads.json")
    parser.add_argument("--base-prs-path", default="fixtures/base_prs.json")
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--classes", nargs="+", default=None,
                         help="restrict to specific attack classes for a focused demo batch")
    parser.add_argument("--skip-payloads", type=int, default=0, help="skip the first N payloads in the deterministic sample (resume after a partial/interrupted run)")
    parser.add_argument("--mitigation-states", nargs="+", default=list(MITIGATION_STATES.keys()))
    parser.add_argument("--benign-prs-path", default="fixtures/benign_prs.json")
    parser.add_argument("--include-benign", action="store_true")
    parser.add_argument("--canary-value", default=os.environ.get("CANARY_DEPLOY_TOKEN"))
    parser.add_argument("--out-index", default="results/live_runs_index.json")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="pause between PRs to stay under API rate limits")
    args = parser.parse_args()

    if not args.canary_value:
        raise SystemExit(
            "No canary value. Set CANARY_DEPLOY_TOKEN in your local .env to the "
            "SAME value you set as the CANARY_DEPLOY_TOKEN repo secret in GitHub "
            "(Settings -> Secrets and variables -> Actions)."
        )

    gh = get_client()
    repo = get_repo(gh)

    payloads = load_payloads(args.payloads_path)
    base_prs = load_base_prs(args.base_prs_path)
    sample = stratified_sample(payloads, args.per_class, classes=args.classes)
    sample = sample[args.skip_payloads:]

    out_path = Path(args.out_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []

    def record(entry: dict):
        # Flush after every PR, not just at the end of the run -- a crash
        # or network blip mid-loop must never lose record of PRs that are
        # already live on GitHub (they can't be scored without this).
        existing.append(entry)
        out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    opened_this_run = 0
    run_i = args.skip_payloads
    for payload in sample:
        base_pr = base_prs[run_i % len(base_prs)]
        for mitigation_state in args.mitigation_states:
            run_i += 1
            pr = open_one_pr(repo, payload, base_pr, mitigation_state, run_i, args.canary_value)
            record({
                "pr_number": pr.number, "pr_url": pr.html_url,
                "payload_id": payload["id"], "class_id": payload["class_id"],
                "goal": payload["goal"], "surface": payload["surface"],
                "base_pr_id": base_pr.get("id", "unknown"),
                "mitigation_state": mitigation_state,
                "is_malicious": True,
                "true_severity": base_pr["true_severity"],
            })
            opened_this_run += 1
            print(f"[open_prs] opened PR #{pr.number} ({payload['class_id']}, {mitigation_state})")
            time.sleep(args.sleep_seconds)

    if args.include_benign:
        benign_prs = json.loads(Path(args.benign_prs_path).read_text(encoding="utf-8"))
        for b in benign_prs:
            for mitigation_state in args.mitigation_states:
                run_i += 1
                pr = open_one_benign_pr(repo, b, mitigation_state, run_i)
                record({
                    "pr_number": pr.number, "pr_url": pr.html_url,
                    "payload_id": "BENIGN_CONTROL", "class_id": "BENIGN_CONTROL",
                    "goal": "BENIGN_CONTROL", "surface": "NONE",
                    "base_pr_id": b.get("id", "unknown"),
                    "mitigation_state": mitigation_state,
                    "is_malicious": False,
                    "true_severity": b.get("true_severity", "low"),
                })
                opened_this_run += 1
                print(f"[open_prs] opened benign control PR #{pr.number} ({mitigation_state})")
                time.sleep(args.sleep_seconds)

    print(f"\n[open_prs] opened/recovered {opened_this_run} PRs this run (index now has {len(existing)} total)")
    print(f"[open_prs] estimated GitHub Actions minutes for this batch: ~{opened_this_run} (1 provider x ~1 min/run)")


if __name__ == "__main__":
    main()
