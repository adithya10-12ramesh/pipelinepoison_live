# PipelinePoison

A systematic taxonomy and mitigation evaluation of prompt-injection attacks
on LLM-integrated DevOps pipelines. Gemini-only build.

## What's in this repo

```
pipelinepoison/
├── harness/
│   ├── taxonomy.py          # frozen 6 goals x 4 surfaces = 24 classes
│   ├── llm_caller.py         # GeminiCaller + MockCaller, rate limit, backoff, JSONL log
│   ├── context_assembler.py  # injects a payload into a base PR's surface
│   ├── judge.py               # rule-based + LLM-as-Judge, disagreement flag
│   ├── results_store.py      # append-only storage + aggregation with CIs
│   └── experiment_runner.py  # orchestrates a full run
├── mitigations/
│   ├── input_filter.py
│   ├── prompt_hardening.py
│   ├── output_validator.py
│   └── spotlighting.py       # spotlighting + sandwich defences
├── payloads/
│   ├── generate_payloads.py  # regenerates payloads.json deterministically
│   └── payloads.json         # 60 payloads, all 24 classes covered
├── fixtures/
│   └── base_prs.json         # 3 seed base PRs (title, body, diff, ci_log)
├── paper/
│   └── threat_model.md       # Phase 2 deliverable
├── tests/
│   └── test_smoke.py         # full offline test suite, no API key needed
├── .github/workflows/pipelinepoison.yml
├── run_experiment.py         # single CLI entry point (FR8)
├── requirements.txt
└── .env.example
```

## What's new in v2 (efficiency + stronger mitigations)

**Efficiency**
- `--skip-existing`: skips cells already stored without an error, so reruns and multi-day resumes never duplicate work.
- `--judge-sample 0.2`: LLM judge on a reproducible 20% of cells (rule-based verdict is always computed). Cuts calls by ~40% while keeping a sample for rule/LLM agreement.
- `--dry-run`: prints the estimated call count and exits without calling the API.
- Circuit breaker: the run stops after 3 consecutive API errors instead of burning time on retries.
- `python check_mistral.py`: 15-second diagnostic for Mistral 403/429 problems.

**Mitigations** (use with `--mitigation-states`)
| State | What it does |
|---|---|
| `none` | baseline (note: the base system prompt already says to ignore instructions in PR text) |
| `input_filter_only` | v2 filter: Unicode/zero-width normalisation + 7 generic injection pattern families |
| `prompt_hardening_only` | self-reminder suffix in the system prompt |
| `output_validation_only` | v2 validator: secrets/credential formats, insecure code, out-of-scope actions, unconditional approval |
| `spotlighting_only` | all PR content wrapped in a per-cell delimiter declared untrusted in the system prompt |
| `sandwich_only` | task reminder repeated AFTER the untrusted content |
| `all_combined` | input filter + hardening + output validation |
| `strong_combined` | all five defences together |

Aggregate tables now include `rate_after_validation`: an attack only counts as successful if the output validator did not flag the reply.

**Methodology cautions (put these in your limitations section)**
1. The v2 input-filter patterns were written from general injection taxonomies, NOT tuned to `payloads.json`. Do not tune them on the same payloads you report on; if you iterate, hold out ~20 payloads as a test set.
2. The output validator and the rule-based judge share some patterns, so success-after-validation is optimistic by construction. Report it next to the LLM-judge verdict and the raw rate.
3. Rule-based judge patterns are literal and may under- or over-count. Read ~30 success and ~30 non-success records by hand and quote the agreement rate.

Recommended collection command (resumable, cheaper):
```bash
python run_experiment.py --provider gemini --skip-existing --judge-sample 0.2 \
  --mitigation-states none prompt_hardening_only spotlighting_only sandwich_only strong_combined --offset 0 --limit 10
```

## 1. One-time setup

```bash
cd pipelinepoison
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Get a Gemini API key from https://aistudio.google.com/apikey and put it in
`.env`:

```
GEMINI_API_KEY=your-real-key-here
```

**Do this before anything else**: `.env` is already in `.gitignore`. Never
commit it. If you ever push a key by accident, revoke it immediately in
Google AI Studio and rotate.

## 2. Verify everything works — offline, no API key needed

```bash
pytest tests/ -v
```

You should see 20 tests pass. This proves the taxonomy, payload corpus,
context assembler, all three mitigations, the judge, and the full
experiment runner all work correctly, using a deterministic `MockCaller`
that makes zero network calls.

Then run the CLI itself against the mock provider:

```bash
python run_experiment.py --mock --limit 4
```

This exercises the exact same code path a real Gemini run uses, and writes
real result files to `results/`.

## 3. Run a small real smoke test against Gemini (costs a handful of calls)

```bash
python run_experiment.py --provider gemini --limit 3 --mitigation-states none
```

This runs 3 payloads x 3 base PRs x 1 mitigation state = 9 main calls + 9
judge calls = 18 calls total. Check `results/verdicts.jsonl` and the
printed aggregate table.

## 4. Run the full evaluation (all 60 payloads)

```bash
python run_experiment.py --provider gemini
```

Default config: all 60 payloads x 3 base PRs x 2 mitigation states
(`none`, `all_combined`) x 2 calls per cell (main + judge) =
**720 API calls**. At the free tier's default 10 requests/minute that's
about **72 minutes** — *if* your project's per-day quota is high enough to
sustain it. Some free-tier projects are capped much lower (as low as
20 requests/day for `gemini-3.5-flash` on some projects) — check your
live quota at https://ai.dev/rate-limit before assuming the per-minute
number is what actually gates you.

### If you're on a low daily quota (staying within the free tier)

Two levers bring the call count down to fit a tight daily cap without
paying anything:

**A. Skip the LLM-judge during the main sweep** (`--no-llm-judge`), which
halves calls-per-cell from 2 to 1. Run a smaller, separate LLM-judge-only
pass afterward on a stratified sample to measure rule-based/LLM-judge
agreement — a legitimate reduced-cost design, not a corner cut, and it
maps directly onto the "judge fragility" discussion already in
`paper/threat_model.md` and the lit survey.

**B. Spread the sweep across multiple days using `--offset` to resume**
without re-running payloads you've already covered:

```bash
# Day 1 (20 payloads at ~1 call/payload/base-PR with --no-llm-judge)
python run_experiment.py --provider gemini --no-llm-judge --offset 0 --limit 20

# Day 2 — the previous run's output tells you the next --offset to use
python run_experiment.py --provider gemini --no-llm-judge --offset 20 --limit 20

# ...continue until offset reaches 60 (all payloads covered)
```

Every run appends to the same `results/verdicts.jsonl`, so the aggregate
tables accumulate correctly across days — no manual merging needed. Each
run's final output prints the exact `--offset` to use next.

To keep the run's cost-tracking metadata honest across a multi-day
collection (NFR1 reproducibility), note the date range you actually ran
in your methodology section rather than treating it as a single sitting.

### If you'd rather just pay (very cheap)

At published `gemini-3.5-flash` paid-tier rates (~$1.50/1M input tokens,
~$9.00/1M output tokens), a full 720-call run costs **roughly $1**, even
with a generous margin. Enabling billing on your Google Cloud project
(Settings → Billing in Google AI Studio) typically also removes the harsh
per-day free-tier cap entirely. Either path — free-tier spread across
days, or a few dollars of billing — gets you full taxonomy coverage before
the deadline; pick whichever fits your constraints.

To measure each mitigation's *individual* marginal delta (not just
combined), run:

```bash
python run_experiment.py --provider gemini --mitigation-states none input_filter_only prompt_hardening_only output_validation_only all_combined
```
This multiplies the call count by 5/2 relative to the default — budget
accordingly (NFR4: the harness will refuse to start if the estimate
exceeds `MAX_CALLS_PER_RUN` in `.env`).

## 5. What each run produces

- `results/verdicts.jsonl` — one record per (payload, base PR, mitigation
  state) cell: both verdicts, disagreement flag, latency, token counts.
- `results/llm_calls.jsonl` — one record per raw API call (secrets
  redacted), for reproducibility (NFR1) and cost auditing.
- `results/aggregate_per_class.csv`
- `results/aggregate_per_mitigation.csv`
- `results/aggregate_per_model.csv`

These three CSVs are the tables your Results/Discussion sections need —
per-class attack success rate, per-mitigation marginal delta, each with a
95% confidence interval (Wilson score interval, robust at small n).

## 6. GitHub Actions

`.github/workflows/pipelinepoison.yml` runs the same harness inside CI,
either on a real pull request or via manual `workflow_dispatch`. Add your
key as a repository secret named `GEMINI_API_KEY` (Settings → Secrets and
variables → Actions → New repository secret) before triggering it —
**never put the key in the workflow file itself.**

## 7. Regenerating or extending the payload corpus

```bash
python payloads/generate_payloads.py
```

This is deterministic — same output every time — because it's meant to be
re-run after you edit `TEMPLATES` in that file (e.g. to add literature-
derived payloads from new papers you deep-read). The script will refuse to
finish if any of the 24 classes ends up with zero payloads, so you can't
accidentally break taxonomy coverage.

## 8. Adding a second model — Mistral is already wired in

Per NFR5, the Mistral integration only required:
1. Subclassing `LLMCaller` in `harness/llm_caller.py` → `MistralCaller`, implementing `_invoke()` and `_transient_exceptions()`
2. One line in `_PROVIDER_REGISTRY`
3. New env vars in `.env.example` (`MISTRAL_API_KEY`, `MISTRAL_MODEL`, `MISTRAL_RATE_LIMIT_RPM`)
4. `mistral` added as a `--provider` choice in `run_experiment.py`

No changes needed to `context_assembler.py`, `judge.py`, `results_store.py`, or `experiment_runner.py` — this is exactly the extensibility contract the harness was designed around.

To use it:
1. Get a free key at https://console.mistral.ai (no card required at time of writing — verify current terms yourself)
2. Add `MISTRAL_API_KEY=...` to your `.env`
3. Run:
   ```bash
   python run_experiment.py --provider mistral --limit 3
   ```

**Note on `mistralai` SDK versions:** the client class location changed between SDK major versions (`mistralai.Mistral` in 1.x, `mistralai.client.Mistral` in 2.x). The import in `llm_caller.py` tries both automatically, so `pip install -r requirements.txt` should work regardless of which version pip resolves. This integration has been verified to construct and call correctly against SDK version 2.9.4's method signatures, but the actual live API call hasn't been tested against a real key — run the `--limit 3` smoke test above before trusting a full sweep.

Adding a **third** provider (e.g. Groq, OpenAI) follows the identical pattern.
