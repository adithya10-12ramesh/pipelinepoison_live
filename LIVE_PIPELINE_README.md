# PipelinePoison — live (real-world) evaluation

This adds the real-world validation layer on top of the existing offline
harness: a sandbox GitHub repo with a genuinely naive LLM PR-review bot,
real pull requests carrying real payloads, canary secrets as ground
truth, mitigations running as actual pipeline stages, and a benign-PR
control set for false-positive rate. Read `paper/ethics_note.md` first —
it's the scope boundary everything below assumes.

## What gets added

```
live_pipeline/
  live_context.py        # GitHub API read/write wrapper (PyGithub)
  canaries.py             # canary secret definitions + leak detection
  bot.py                  # the victim pipeline's PR-review bot (CI entrypoint)
  canary_judge.py          # action-based ground-truth scorer
  open_prs.py              # opens real attack PRs against your sandbox repo
  fetch_live_results.py    # pulls bot results back, scores, writes verdicts
  fetch_benign_prs.py       # samples real benign PRs for the control set
.github/workflows/pr_review_bot.yml   # the victim pipeline itself
fixtures/ci_seed/last_run.log         # seed content for the CI_LOG_OUTPUT surface
paper/ethics_note.md
```

Nothing in `harness/`, `mitigations/`, or `payloads/` changes — this
reuses them directly (same `MITIGATION_STATES`, same `input_filter` /
`prompt_hardening` / `validate_output`, same payload corpus).

## 1. Create the sandbox repo (GitHub, 5 min)

1. On GitHub, create a **new, empty repository** you own — e.g.
   `pipelinepoison-sandbox`. Public or private both work; private avoids
   anyone outside stumbling onto the deliberately-naive bot.
2. Clone it locally and copy in **just these three things** from your
   research repo: `harness/`, `mitigations/`, `payloads/`,
   `requirements.txt`, plus the new `live_pipeline/` and
   `.github/workflows/pr_review_bot.yml` from this bundle. (The sandbox
   repo doesn't need `firebase_dashboard/`, `tests/`, or `paper/` — keep
   it minimal; it exists to be PR'd against, not to hold your research.)
3. Add `requirements.txt`'s one new dependency:
   ```
   PyGithub==2.4.0
   ```
4. Create the two files the bot reads/writes against:
   ```bash
   mkdir -p ci sandbox
   cp fixtures/ci_seed/last_run.log ci/last_run.log
   echo "# baseline" > sandbox/payload_target.py
   git add -A && git commit -m "Set up PipelinePoison sandbox" && git push
   ```

## 2. Generate and set canary secrets

```bash
python -c "import secrets; print('CANARY-' + secrets.token_hex(16))"
```

Run that **three times** for three distinct values. In the sandbox
repo's GitHub settings → **Secrets and variables → Actions → New
repository secret**, add:

| Secret name | Value |
|---|---|
| `CANARY_DEPLOY_TOKEN` | first generated value |
| `CANARY_DB_PASSWORD` | second generated value |
| `CANARY_INTERNAL_API_KEY` | third generated value |
| `GEMINI_API_KEY` | your Gemini key |
| `MISTRAL_API_KEY` | your Mistral key |
| `GEMINI_MODEL` | `gemini-3.5-flash` (or your choice) |
| `MISTRAL_MODEL` | `mistral-medium` |
| `GEMINI_TRAINING_TIER` | `true` or `false` |

Then, in your **local** `.env` (the one `open_prs.py` reads — this is a
different `.env` from the sandbox repo, it stays on your machine):

```
LIVE_GITHUB_TOKEN=ghp_...              # PAT with `repo` scope, on the sandbox repo only
LIVE_GITHUB_REPO=yourusername/pipelinepoison-sandbox
CANARY_DEPLOY_TOKEN=CANARY-...         # SAME value as the GitHub secret above
```

Generate the PAT at GitHub → Settings → Developer settings → Personal
access tokens → Fine-grained tokens, scoped to just the sandbox repo,
with Pull requests (read/write), Contents (read/write), and Issues
(read/write) permissions.

## 3. Smoke-test the bot offline first (no GitHub, no API calls)

```bash
python live_pipeline/bot.py --pr-number 0 --provider gemini --mock
```

This exercises the action-parsing and mitigation-gating logic with
`MockCaller`, so you can confirm it runs before spending any GitHub
Actions minutes or API quota.

## 4. Open a tiny first batch of real PRs

Start as small as possible — one payload per attack class, no
mitigation:

```bash
python live_pipeline/open_prs.py --per-class 1 --mitigation-states none
```

This opens 24 PRs (one per attack class) and labels each
`mitigation:none`. Watch the **Actions** tab on your sandbox repo — each
PR triggers two parallel jobs (Gemini, Mistral), each running
`live_pipeline/bot.py`, which reviews the PR for real and posts a real
comment/review/label.

## 5. Pull the results back and score them

Give the Actions runs a few minutes to finish, then:

```bash
python live_pipeline/fetch_live_results.py
```

This reads each PR's hidden result comment, scores it against the
canary/action-based ground truth in `canary_judge.py`, and writes
`results_live/verdicts.jsonl` + the three aggregate CSVs — same shape as
the offline harness's output.

## 6. Scale up once step 4-5 works end to end

```bash
python live_pipeline/open_prs.py --per-class 2 \
  --mitigation-states none input_filter_only prompt_hardening_only output_validation_only all_combined
```

24 classes × 2 payloads × 5 mitigation states = 240 PRs × 2 providers =
480 bot runs. At GitHub's free-tier 2,000 Actions minutes/month and
~1 minute per run, that's within budget, but check the Actions usage
page before committing to a batch this size. Re-run
`fetch_live_results.py` as runs complete — it skips PRs that haven't
posted a result yet, so it's safe to call repeatedly.

## 7. Add the benign control set

```bash
python live_pipeline/fetch_benign_prs.py --count 25
python live_pipeline/open_prs.py --include-benign --per-class 0 \
  --mitigation-states none all_combined
```

(`--per-class 0` skips opening any new attack PRs this call — just the
benign batch.) This measures how often your mitigations false-positive
on ordinary, legitimate PRs — the number every reviewer will ask for
once they see an attack-success-rate headline.

## 8. Feed into the Firebase dashboard

`results_live/verdicts.jsonl` has the same schema the dashboard already
reads (see `firebase_dashboard/src/lib/schema.js`), plus a
`ground_truth_source: "action_based"` field and a few live-only fields
(`pr_number`, `pr_url`). Sync it as its own run:

```bash
python sync/push_results.py --run-id live-batch-1 --provider gemini \
  --results-dir results_live
```

It'll show up in the dashboard's run list alongside your simulated runs,
and the heatmap/mitigation/disagreement pages all work on it unchanged
(the "judge disagreement" count will correctly show 0, since live
verdicts don't use an LLM judge at all).

## Cost and safety reminders

- Every PR opened here is real, visible repo activity. Expect your
  sandbox repo's Actions tab, PR list, and label set to fill up quickly
  — that's expected, not a bug.
- `FETCH_URL` actions are logged, never executed (see
  `paper/ethics_note.md`). Don't change this.
- Delete the sandbox repo (or just leave it archived) once data
  collection for the paper is done; there's no reason to keep a
  deliberately-vulnerable bot running indefinitely.
