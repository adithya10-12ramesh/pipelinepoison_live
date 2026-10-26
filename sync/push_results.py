#!/usr/bin/env python3
"""
push_results.py

Uploads PipelinePoison results (verdicts.jsonl, llm_calls.jsonl, and the
three aggregate CSVs) into Firestore, so the dashboard can read them.

Run this after any `run_experiment.py` run:

    python sync/push_results.py --run-id 2026-10-03-smoke

    # a differently-named results file, with a label for the dashboard:
    python sync/push_results.py --run-id 2026-10-06-ollama \
        --verdicts-path results/ollama_ministral3b.jsonl --payload-set mixed30

    # preview the summary without writing anything:
    python sync/push_results.py --run-id test --dry-run

Setup (one-time):
  1. Firebase console -> Project settings -> Service accounts ->
     Generate new private key. Save it as `firebase-service-account.json`
     in the repo root (it's already covered by a .gitignore entry you
     should add — NEVER commit this file).
  2. pip install firebase-admin
  3. Set FIREBASE_PROJECT_ID in .env (or pass --project).

Firestore layout:

  runs/{run_id}
    - provider, task, mitigation_states, started_at, finished_at,
      total_calls, status

  runs/{run_id}/verdicts/{verdict_id}
    - one doc per (payload, base_pr, mitigation_state) cell, same fields
      as a line in verdicts.jsonl

  runs/{run_id}/llm_calls/{call_id}
    - one doc per raw API call (secrets already redacted upstream by
      your harness's JSONL logger — this script does not re-redact)

  runs/{run_id}/aggregates/per_class
  runs/{run_id}/aggregates/per_mitigation
  runs/{run_id}/aggregates/per_model
    - each the corresponding CSV, stored as a `rows: [...]` array

This script is append-only per run_id: re-running it with the same
--run-id overwrites that run's docs (safe to re-sync after a resumed
multi-day run finishes).
"""

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    print("Missing dependency. Run: pip install firebase-admin", file=sys.stderr)
    sys.exit(1)

BATCH_LIMIT = 400  # Firestore batch hard limit is 500 writes; stay under it


def init_firestore(project_id: str | None, cred_path: str):
    if not os.path.exists(cred_path):
        print(
            f"Service account file not found at {cred_path}.\n"
            "Download it from Firebase console -> Project settings -> "
            "Service accounts -> Generate new private key.",
            file=sys.stderr,
        )
        sys.exit(1)
    cred = credentials.Certificate(cred_path)
    kwargs = {"projectId": project_id} if project_id else {}
    firebase_admin.initialize_app(cred, kwargs)
    return firestore.client()


def read_jsonl(path):
    if not os.path.exists(path):
        print(f"  (skipping, not found: {path})")
        return []
    records = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                # A log file an interrupted process was still appending to
                # can end in a partial/corrupted line -- skip it rather than
                # crashing the whole push over one bad record.
                skipped += 1
                print(f"  WARNING: skipping corrupted line {i} in {path}: {e}")
    if skipped:
        print(f"  ({skipped} corrupted line(s) skipped in {path})")
    return records


def read_csv_rows(path):
    if not os.path.exists(path):
        print(f"  (skipping, not found: {path})")
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def batched_write(db, docs_with_refs, label):
    """docs_with_refs: list of (doc_ref, data_dict)"""
    total = len(docs_with_refs)
    if total == 0:
        print(f"  {label}: 0 docs")
        return
    for i in range(0, total, BATCH_LIMIT):
        chunk = docs_with_refs[i : i + BATCH_LIMIT]
        batch = db.batch()
        for ref, data in chunk:
            batch.set(ref, data)
        batch.commit()
    print(f"  {label}: {total} docs")


# --------------------------------------------------------------------------- #
# Dashboard support: payload text + run summary
# --------------------------------------------------------------------------- #

VALIDATOR_STATES = {"output_validation_only", "all_combined", "strong_combined"}


def load_payload_texts(paths):
    """payload_id -> (text, source file). Earlier files win on duplicate ids."""
    texts = {}
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  WARNING: could not read payload file {path}: {e}")
            continue
        if not isinstance(data, list):
            continue
        for p in data:
            if isinstance(p, dict) and p.get("id") and p.get("text"):
                texts.setdefault(p["id"], (p["text"], os.path.basename(path)))
    return texts


def _wilson(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return p, max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom)


def summarise(verdicts):
    """Headline numbers for run_index (shown on the dashboard's Runs page).
    Same definitions as the dashboard: errored records are excluded; raw =
    rule_based_success; effective = raw AND NOT output_validation_flagged."""
    valid = [v for v in verdicts if not v.get("response_error")]
    by_state = {}
    for v in valid:
        s = by_state.setdefault(v.get("mitigation_state", "unknown"), {"n": 0, "raw": 0, "eff": 0})
        s["n"] += 1
        if v.get("rule_based_success") is True:
            s["raw"] += 1
            if v.get("output_validation_flagged") is not True:
                s["eff"] += 1

    order = ["none", "input_filter_only", "prompt_hardening_only", "output_validation_only",
             "spotlighting_only", "sandwich_only", "all_combined", "strong_combined"]
    states = sorted(by_state, key=lambda s: (order.index(s) if s in order else 99, s))
    baseline = "none" if "none" in by_state else (states[0] if states else None)

    out = {
        "mitigation_states": states,
        "error_count": len(verdicts) - len(valid),
        "payload_count": len({v.get("payload_id") for v in valid if v.get("payload_id")}),
        "base_pr_count": len({v.get("base_pr_id") for v in valid if v.get("base_pr_id")}),
        "models": sorted({v.get("model") for v in valid if v.get("model")}),
        "baseline_state": baseline,
        "baseline_asr": None,
        "best_effective_asr": None,
        "best_state": None,
        "brief": None,
    }
    if not baseline:
        return out

    b = by_state[baseline]
    base_rate, base_low, base_high = _wilson(b["raw"], b["n"])
    out["baseline_asr"] = round(base_rate, 4)

    best = None
    for s in states:
        if s == baseline:
            continue
        rate = by_state[s]["eff"] / by_state[s]["n"]
        if best is None or rate < best[1]:
            best = (s, rate)

    parts = []
    if best:
        out["best_state"], out["best_effective_asr"] = best[0], round(best[1], 4)
        parts.append(
            f"{best[0]} cuts effective ASR from {base_rate * 100:.1f}% to {best[1] * 100:.1f}%."
        )
    flat = [
        s for s in states
        if s != baseline and s not in VALIDATOR_STATES
        and base_low <= by_state[s]["raw"] / by_state[s]["n"] <= base_high
    ]
    if flat:
        parts.append(f"{', '.join(flat)} stay within the baseline CI.")
    out["brief"] = " ".join(parts) or None
    return out


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Push PipelinePoison results to Firestore.")
    parser.add_argument("--run-id", required=True, help="unique id for this run, e.g. 2026-10-03-smoke")
    parser.add_argument("--project", default=os.environ.get("FIREBASE_PROJECT_ID"))
    parser.add_argument(
        "--credentials",
        default=os.environ.get("FIREBASE_CREDENTIALS", "firebase-service-account.json"),
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--verdicts-path", default=None,
        help="results file to push (default: <results-dir>/verdicts.jsonl), "
             "e.g. results/ollama_ministral3b.jsonl",
    )
    parser.add_argument(
        "--calls-path", default=None,
        help="LLM call log (default: <verdicts>_calls.jsonl if it exists, else <results-dir>/llm_calls.jsonl)",
    )
    parser.add_argument(
        "--payloads", nargs="*", default=None,
        help="payload JSON files used to attach payload_text to each verdict (default: payloads/*.json)",
    )
    parser.add_argument("--payload-set", default=None, help="label shown on the dashboard, e.g. mixed30")
    parser.add_argument("--provider", default=None, help="default: read from the verdict records")
    parser.add_argument("--task", default="review")
    parser.add_argument(
        "--mitigation-states", nargs="+", default=None,
        help="default: the states actually present in the verdict records",
    )
    parser.add_argument("--status", default="complete", choices=["running", "complete", "failed"])
    parser.add_argument("--dry-run", action="store_true", help="print what would be pushed, write nothing")
    args = parser.parse_args()

    # ── Resolve input files ────────────────────────────────────────────
    default_verdicts = os.path.join(args.results_dir, "verdicts.jsonl")
    verdicts_path = args.verdicts_path or default_verdicts
    stem = os.path.splitext(verdicts_path)[0]
    if args.calls_path:
        calls_path = args.calls_path
    elif args.verdicts_path and os.path.exists(f"{stem}_calls.jsonl"):
        calls_path = f"{stem}_calls.jsonl"
    else:
        calls_path = os.path.join(args.results_dir, "llm_calls.jsonl")
    # run_experiment.py writes a non-default results file's CSVs to a folder
    # named after it (results/foo.jsonl -> results/foo/).
    csv_dir = stem if args.verdicts_path and os.path.isdir(stem) else args.results_dir
    per_class_path = os.path.join(csv_dir, "aggregate_per_class.csv")
    per_mitigation_path = os.path.join(csv_dir, "aggregate_per_mitigation.csv")
    per_model_path = os.path.join(csv_dir, "aggregate_per_model.csv")

    print(f"Syncing run '{args.run_id}' from {verdicts_path} ...")

    verdicts = read_jsonl(verdicts_path)
    calls = read_jsonl(calls_path)
    per_class = read_csv_rows(per_class_path)
    per_mitigation = read_csv_rows(per_mitigation_path)
    per_model = read_csv_rows(per_model_path)

    # ── Attach payload text so the Justifications tab can show it ──────
    payload_files = args.payloads if args.payloads is not None else sorted(glob.glob("payloads/*.json"))
    texts = load_payload_texts(payload_files)
    attached = 0
    for v in verdicts:
        hit = texts.get(v.get("payload_id"))
        if hit:
            v["payload_text"], v["payload_source"] = hit
            attached += 1
    print(f"  payload text attached to {attached}/{len(verdicts)} verdicts (from {len(payload_files)} file(s))")

    # ── Summary for the Runs page ──────────────────────────────────────
    summary = summarise(verdicts)
    providers = sorted({v.get("provider") for v in verdicts if v.get("provider")})
    provider = args.provider or (", ".join(providers) if providers else "unknown")
    mitigation_states = args.mitigation_states or summary["mitigation_states"]
    disagreements = sum(1 for v in verdicts if v.get("judge_disagreement"))
    now = datetime.now(timezone.utc).isoformat()

    index_doc = {
        "run_id": args.run_id,
        "provider": provider,
        "model": ", ".join(summary["models"]) or None,
        "payload_set": args.payload_set,
        "task": args.task,
        "status": args.status,
        "synced_at": now,
        "verdict_count": len(verdicts),
        "disagreement_count": disagreements,
        "error_count": summary["error_count"],
        "payload_count": summary["payload_count"],
        "base_pr_count": summary["base_pr_count"],
        "mitigation_states": mitigation_states,
        "baseline_state": summary["baseline_state"],
        "baseline_asr": summary["baseline_asr"],
        "best_effective_asr": summary["best_effective_asr"],
        "best_state": summary["best_state"],
        "brief": summary["brief"],
    }

    if args.dry_run:
        print("\n[dry run] run_index document that would be written:")
        print(json.dumps(index_doc, indent=2))
        print(f"[dry run] {len(verdicts)} verdicts, {len(calls)} calls — nothing written.")
        return

    db = init_firestore(args.project, args.credentials)
    run_ref = db.collection("runs").document(args.run_id)

    # Run summary doc
    # Your verdict records have no explicit "id" (see harness/experiment_runner.py's
    # `record = {...}` block) — disagreement is the "judge_disagreement" key, and
    # success is "rule_based_success" (the primary metric your own results_store.py
    # uses, since it's always available even when the judge call fails/is skipped).
    run_ref.set(
        {
            **{k: index_doc[k] for k in (
                "provider", "model", "payload_set", "task", "mitigation_states", "status",
                "synced_at", "verdict_count", "disagreement_count", "error_count",
            )},
            "call_count": len(calls),
            "source_file": verdicts_path.replace("\\", "/"),
        },
        merge=True,
    )

    # Verdicts subcollection. Records have no explicit "id", so build a
    # stable composite key from (payload_id, base_pr_id, mitigation_state) —
    # the same triple that uniquely identifies a cell in
    # experiment_runner.py's itertools.product() loop. This keeps doc IDs
    # stable across re-syncs of a resumed/extended run, so adjudications
    # recorded against a verdict (see Disagreements.jsx) don't get orphaned.
    verdict_docs = []
    for i, v in enumerate(verdicts):
        vid = v.get("id") or "__".join(
            str(v.get(k, "?")) for k in ("payload_id", "base_pr_id", "mitigation_state")
        )
        v = {**v, "id": vid}
        ref = run_ref.collection("verdicts").document(vid)
        verdict_docs.append((ref, v))
    batched_write(db, verdict_docs, "verdicts")

    # llm_calls subcollection
    call_docs = []
    for i, c in enumerate(calls):
        cid = c.get("id") or f"c{i:05d}"
        ref = run_ref.collection("llm_calls").document(str(cid))
        call_docs.append((ref, c))
    batched_write(db, call_docs, "llm_calls")

    # Aggregates (one doc each, holding the full table — these are small)
    run_ref.collection("aggregates").document("per_class").set({"rows": per_class})
    run_ref.collection("aggregates").document("per_mitigation").set({"rows": per_mitigation})
    run_ref.collection("aggregates").document("per_model").set({"rows": per_model})
    print(f"  aggregates: per_class={len(per_class)} per_mitigation={len(per_mitigation)} per_model={len(per_model)}")

    # Flat top-level index for the dashboard's Runs page (now with headline numbers)
    db.collection("run_index").document(args.run_id).set(index_doc)

    print(f"Done. View at your dashboard URL -> run '{args.run_id}'.")


if __name__ == "__main__":
    main()
