"""
harness/experiment_runner.py

ExperimentRunner ties PayloadCorpus + BasePR pool + ContextAssembler +
LLMCaller + Judge + ResultsStore together into the single run script
required by FR8. Given a corpus file and a model, produces a complete
result table without further human intervention, and records enough
metadata for a third party to reproduce it (NFR1).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .context_assembler import BasePR, render_context
from .judge import evaluate
from .llm_caller import LLMCaller
from .results_store import ResultsStore
from .taxonomy import Surface

from mitigations.input_filter import input_filter
from mitigations.prompt_hardening import harden_prompt
from mitigations.output_validator import validate_output
from mitigations.spotlighting import spotlight, sandwich


MITIGATION_STATES = {
    "none": {"input_filter": False, "prompt_hardening": False, "output_validation": False},
    "input_filter_only": {"input_filter": True, "prompt_hardening": False, "output_validation": False},
    "prompt_hardening_only": {"input_filter": False, "prompt_hardening": True, "output_validation": False},
    "output_validation_only": {"input_filter": False, "prompt_hardening": False, "output_validation": True},
    "all_combined": {"input_filter": True, "prompt_hardening": True, "output_validation": True},
    # Stronger prompt-structure defences (mitigations/spotlighting.py)
    "spotlighting_only": {"spotlighting": True},
    "sandwich_only": {"sandwich": True},
    "strong_combined": {
        "input_filter": True, "prompt_hardening": True, "output_validation": True,
        "spotlighting": True, "sandwich": True,
    },
}


def load_payloads(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class RunConfig:
    payloads_path: str = "payloads/payloads.json"
    base_prs_path: str = "fixtures/base_prs.json"
    task: str = "review"
    mitigation_states: tuple[str, ...] = ("none", "all_combined")
    max_calls: Optional[int] = None
    limit_payloads: Optional[int] = None  # for smoke tests / cost control
    offset_payloads: int = 0  # for resuming a multi-day run (skip N payloads already covered)
    run_llm_judge: bool = True
    # Skip cells already stored without an API error (payload, PR, state, model, task).
    skip_existing: bool = False
    # Fraction of cells (0-1) that also get the LLM-judge call. The rule-based
    # verdict is always computed. Sampling cuts judge calls while still giving a
    # reproducible subset for measuring rule/LLM agreement.
    judge_sample_rate: float = 1.0
    dry_run: bool = False  # print the call estimate and exit without calling the API
    # Circuit breaker: stop the run after this many API errors in a row. A
    # sustained error (quota, auth, tier, outage) will not fix itself mid-run,
    # and every failed cell costs a full retry/backoff cycle of wall-clock time.
    max_consecutive_errors: int = 3


def estimate_call_count(
    n_payloads: int, n_mitigation_states: int, run_llm_judge: bool, judge_sample_rate: float = 1.0
) -> int:
    """NFR4: pre-estimate cost per run before making any calls.
    With judge sampling, only a fraction of cells make the extra judge call."""
    calls_per_cell = 1 + (judge_sample_rate if run_llm_judge else 0)  # main call + (sampled) judge call
    return int(round(n_payloads * n_mitigation_states * calls_per_cell))


def _in_judge_sample(payload_id: str, base_pr_id: str, state: str, rate: float) -> bool:
    """Deterministic, reproducible sampling: the same cell is always in or out
    of the LLM-judge sample, so reruns and resumed runs stay consistent."""
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    h = int(hashlib.sha256(f"{payload_id}|{base_pr_id}|{state}".encode()).hexdigest()[:8], 16)
    return (h % 10000) < int(rate * 10000)


def run_experiment(
    caller: LLMCaller,
    config: RunConfig,
    judge_caller: Optional[LLMCaller] = None,
    store: Optional[ResultsStore] = None,
) -> ResultsStore:
    store = store or ResultsStore()
    judge_caller = judge_caller or caller  # same model can judge itself, flagged as a limitation

    payloads = load_payloads(config.payloads_path)
    if config.offset_payloads:
        payloads = payloads[config.offset_payloads:]
    if config.limit_payloads:
        payloads = payloads[: config.limit_payloads]
    base_prs = BasePR.load_all(config.base_prs_path)

    n_calls_estimate = estimate_call_count(
        len(payloads) * len(base_prs), len(config.mitigation_states), config.run_llm_judge,
        config.judge_sample_rate,
    )
    max_calls = config.max_calls
    if max_calls and n_calls_estimate > max_calls:
        raise RuntimeError(
            f"Estimated {n_calls_estimate} calls exceeds MAX_CALLS_PER_RUN={max_calls}. "
            "Reduce payload count, base PR count, or mitigation states, or raise the ceiling."
        )
    print(f"[ExperimentRunner] estimated calls for this run: {n_calls_estimate}")
    if config.dry_run:
        already = len(store.completed_keys()) if config.skip_existing else 0
        print(f"[ExperimentRunner] dry run: no API calls made ({already} completed cells would be skipped).")
        return store

    run_meta = {
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "model": caller.model_name,
        "provider": caller.provider_name,
        "training_tier": caller.training_tier,
    }
    print(f"[ExperimentRunner] run metadata: {run_meta}")

    total = len(payloads) * len(base_prs) * len(config.mitigation_states)
    done = 0
    quota_exhausted = False
    consecutive_errors = 0
    skipped = 0
    already_done = store.completed_keys() if config.skip_existing else set()

    for payload, base_pr, mitigation_state in itertools.product(payloads, base_prs, config.mitigation_states):
        if quota_exhausted:
            break
        done += 1
        if already_done and (
            payload["id"], base_pr.id, mitigation_state, caller.model_name, config.task
        ) in already_done:
            skipped += 1
            continue
        flags = {k: False for k in (
            "input_filter", "prompt_hardening", "output_validation", "spotlighting", "sandwich"
        )}
        flags.update(MITIGATION_STATES[mitigation_state])

        ctx = render_context(
            base_pr=base_pr,
            surface=Surface(payload["surface"]),
            payload_text=payload["text"],
            task=config.task,
            input_filter=input_filter if flags["input_filter"] else None,
        )
        system_prompt = ctx["system_prompt"]
        user_prompt = ctx["user_prompt"]
        if flags["prompt_hardening"]:
            system_prompt = harden_prompt(system_prompt)
        if flags["spotlighting"]:
            system_prompt, user_prompt = spotlight(
                system_prompt, user_prompt, seed=f"{payload['id']}|{base_pr.id}"
            )
        if flags["sandwich"]:
            user_prompt = sandwich(user_prompt, config.task)

        response = caller.call(
            user_prompt,
            system_prompt=system_prompt,
            run_metadata={
                "payload_id": payload["id"],
                "class_id": payload["class_id"],
                "base_pr_id": base_pr.id,
                "mitigation_state": mitigation_state,
                "task": config.task,
            },
        )

        if response.error and "perday" in response.error.lower().replace(" ", "").replace("_", ""):
            print(
                f"\n[ExperimentRunner] STOPPING EARLY at cell {done}/{total}: "
                f"daily API quota exhausted. This is a per-DAY limit, not "
                f"per-minute — retrying now won't help. Resume tomorrow, or "
                f"enable billing on your Google Cloud project for higher "
                f"limits. Partial results up to this point are saved in "
                f"{store.path}.\n"
            )
            quota_exhausted = True
            break

        output_flagged, output_flag_reason = (False, None)
        if flags["output_validation"] and not response.error:
            output_flagged, output_flag_reason = validate_output(response.text)

        verdict = evaluate(
            goal=payload["goal"],
            response_text=response.text,
            judge_caller=judge_caller if (
                config.run_llm_judge
                and not response.error
                and _in_judge_sample(payload["id"], base_pr.id, mitigation_state, config.judge_sample_rate)
            ) else None,
        )

        record = {
            "payload_id": payload["id"],
            "class_id": payload["class_id"],
            "goal": payload["goal"],
            "surface": payload["surface"],
            "base_pr_id": base_pr.id,
            "task": config.task,
            "model": caller.model_name,
            "provider": caller.provider_name,
            "mitigation_state": mitigation_state,
            "response_text": response.text[:2000],  # cap stored size
            "response_error": response.error,
            "latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "rule_based_success": verdict.rule_based_success,
            "rule_based_reason": verdict.rule_based_reason,
            "llm_judge_success": verdict.llm_judge_success,
            "llm_judge_reason": verdict.llm_judge_reason,
            "judge_disagreement": verdict.disagreement,
            "output_validation_flagged": output_flagged,
            "output_validation_reason": output_flag_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        store.append(record)

        if response.error:
            consecutive_errors += 1
            if config.max_consecutive_errors and consecutive_errors >= config.max_consecutive_errors:
                print(
                    f"\n[ExperimentRunner] STOPPING EARLY at cell {done}/{total}: "
                    f"{consecutive_errors} API errors in a row. Last error:\n"
                    f"  {response.error[:300]}\n"
                    f"A sustained error (rate limit, quota, model not allowed on your "
                    f"tier, bad key) will not clear by retrying within this run. Fix "
                    f"the cause, then re-run. Errored records are kept in "
                    f"{store.path} but are excluded from success-rate aggregation.\n"
                )
                break
        else:
            consecutive_errors = 0

        if done % 10 == 0 or done == total:
            print(f"[ExperimentRunner] {done}/{total} cells complete")

    if skipped:
        print(f"[ExperimentRunner] skipped {skipped} cells already completed (--skip-existing)")
    return store
