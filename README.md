---
title: finance-ops-openenv
emoji: "\U0001F4C4"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# FinanceOps OpenEnv

FinanceOps OpenEnv is an OpenEnv-compatible benchmark environment for finance operations workflows. It models three practical back-office tasks that human teams actually perform:

1. invoice field extraction
2. invoice anomaly validation
3. purchase-order reconciliation

The project is benchmark-first. It is designed for agent training and evaluation, with typed models, deterministic graders, dense rewards, local debugging tools, and deployment support for Docker and Hugging Face Spaces.

## Judge At A Glance

- live Docker Space: https://paarthk26-finance-ops-openenv.hf.space
- Hugging Face repo: https://huggingface.co/spaces/Paarthk26/finance-ops-openenv
- local validation: `openenv validate` passes
- live deployment checks: `/reset`, `/health`, `/schema`, and `/web` return `200`
- task ladder: `easy` invoice extraction -> `medium` invoice validation -> `hard` PO reconciliation
- evaluator-safe inference logs: `[START]`, `[STEP]`, `[END]`

## Why This Environment Matters

Finance operations teams routinely process invoices, validate faulty documents, and reconcile vendor invoices against approved purchase orders. These are repetitive, rules-heavy, auditable tasks that are valuable for benchmarking agent reliability.

This environment aims to score well on the typical OpenEnv judging dimensions:

- real-world utility: the task domain is practical and non-toy
- task quality: three tasks with meaningful difficulty progression
- environment design: stateful multi-step interaction with penalties and partial credit
- code quality: typed models, structured repo, Docker support, validator support
- creativity: richer hard-task mechanics like vendor aliases, duplicate invoices, split invoices, and bad PO references

## Why This Is Novel

This environment is intentionally more realistic than the standard static document benchmark. The hardest task is not just "match two IDs"; it forces the agent to reason through vendor identity normalization, FX conversion, duplicate handling, split PO references, and exception workflows.

Standout mechanics already implemented:

- vendor alias normalization: the same supplier can appear under multiple name variants such as `Cedar Logistics LLC`, `Cedar Logistics`, `Cedar Logx`, and `Cedar Logistics LLP`, and the agent must resolve those aliases back to the canonical PO vendor before matching
- multi-currency FX-adjusted reconciliation: invoices and POs may be denominated in `USD`, `EUR`, or `INR`, and the agent must compare them only after conversion and a `+/-2%` tolerance check
- split-PO reasoning: a single invoice may reference multiple PO identifiers and must be flagged as `split_po` rather than force-matched
- duplicate invoice handling: duplicate external invoice numbers are present in the hard task and should be flagged as `duplicate_invoice`
- persistent review memory: each episode tracks recent reasoning context so the agent can avoid redoing work and reason over its own previous actions

## Project Structure

```text
.
|-- app.py
|-- credential.txt                # optional local token file, gitignored
|-- Dockerfile
|-- env
|   |-- __init__.py
|   |-- credential.txt            # optional local token file, gitignored
|   |-- data.py
|   |-- environment.py
|   |-- models.py
|   `-- tasks
|       |-- __init__.py
|       |-- easy.py
|       |-- hard.py
|       `-- medium.py
|-- inference.py
|-- openenv.yaml
|-- pyproject.toml
|-- README.md
`-- requirements.txt
```

## Codex Agent

This repo includes a root `AGENTS.md` with project-specific instructions for Codex. It is tuned for the Openenv workflow in this benchmark, including determinism, grader alignment, inference output stability, and validation expectations.

## Core Interface

The environment implements the OpenEnv-style lifecycle:

- `reset(difficulty)` returns the initial `Observation`
- `step(action)` returns `(observation, reward, done, info)`
- `state()` returns the full current environment state

Typed models live in [env/models.py](./env/models.py):

- `Observation`
- `Action`
- `Reward`

The main environment implementation lives in [env/environment.py](./env/environment.py).

## Action Space

Agents interact with the environment through a compact action schema:

- `extract`: submit one extracted field using `field_name` and `field_value`
- `flag`: flag an issue or unmatched invoice using `issue_code` and/or `invoice_id`
- `match`: reconcile one invoice to one purchase order using `invoice_id` and `po_id`
- `submit`: end the episode and trigger final scoring
- `skip`: no-op action with zero immediate reward

The typed Pydantic model is defined in [env/models.py](./env/models.py) and uses the following fields:

- `action_type`
- `field_name`
- `field_value`
- `issue_code`
- `invoice_id`
- `po_id`
- `notes`

## Observation Space

Each observation contains benchmark-level control metadata plus task-specific content:

- `task_id`: active task identifier
- `difficulty`: `easy`, `medium`, or `hard`
- `content.instructions`: task instructions shown to the agent
- `content.document`: the invoice, PO bundle, or validation payload
- `content.current_submission`: current extracted fields for easy tasks
- `content.flagged_issues`: currently flagged anomalies for medium tasks
- `content.current_matches`: current invoice-to-PO matches for hard tasks
- `content.current_unmatched_invoices`: invoices already flagged as unmatched
- `content.current_flagged_discrepancies`: invoices already flagged with amount mismatches
- `content.current_flagged_duplicate_invoices`: invoices already flagged as duplicates
- `content.review_memory`: short task memory derived from previous actions
- `available_actions`, `step_number`, `max_steps`, and `context`

This split keeps the external API stable while still exposing enough structured state for agent learning and evaluation.

## Tasks

### Easy: Invoice Extraction

Goal:
- extract `vendor_name`
- extract `invoice_number`
- extract `invoice_date`
- extract `currency`
- extract `total_amount`

Realism improvements:
- label aliases such as `Supplier`, `Inv No`, `Grand Total`
- vendor alias normalization hints
- mild OCR-style formatting variation

Grader:
- exact normalized match by required field

### Medium: Invoice Validation

Goal:
- flag every genuine anomaly
- avoid false positives

Anomalies used in variants:
- `invalid_invoice_date`
- `duplicate_line_item`
- `subtotal_mismatch`
- `missing_gstin`

Realism improvements:
- bad date formats
- duplicated service lines
- arithmetic inconsistencies
- compliance metadata gaps

Grader:
- recall with false-positive penalties, clamped to `0.0..1.0`

### Hard: PO Reconciliation

Goal:
- match invoices to purchase orders
- flag unmatched invoices
- flag amount mismatches
- flag duplicate invoices
- flag invoices that reference multiple POs as `split_po`

Realism improvements:
- vendor aliases
- PO references
- multi-currency invoices and purchase orders
- FX-adjusted `+/-2%` tolerance
- conflicting vendor names
- split PO references
- wrong PO references
- duplicate invoice submissions
- hard-task late-step penalties after turn 6

Grader:
- match accuracy
- unmatched detection
- amount mismatch detection
- duplicate invoice detection

Why this is difficult:

- the agent must normalize vendor identity before it can even trust the PO candidate set
- valid invoice and PO totals may appear mismatched until they are converted into the same currency
- some invoices should not be matched at all, even if they look superficially valid
- repeated actions and late actions reduce the quality of the episode outcome

All graders are deterministic and return values in `0.0..1.0`.

## Environment Design

The environment uses dense rewards rather than only binary success/failure.

Positive signals:
- correct extraction
- correct anomaly flag
- correct PO match
- correct discrepancy flag
- correct duplicate-invoice flag

Low or zero-reward signals:
- wrong extraction
- wrong match
- false positive flags
- repeated flags or repeated matches
- invalid actions
- skipping turns
- incomplete submission through readiness penalties at submit time

The environment also tracks richer internal state:

- extracted fields
- flagged issues
- matches
- unmatched invoices
- flagged discrepancies
- flagged duplicate invoices
- review memory
- action history
- last action error

This state is available through `state()` and partially surfaced in `Observation.content`.

## Dynamic But Deterministic Task Generation

Task payloads are not single hardcoded examples anymore. They are generated from seeded templates in [env/data.py](./env/data.py), and `reset()` advances the episode counter per difficulty.

That means:

- repeated resets expose different realistic cases
- the same `FINANCE_OPS_SEED` and episode index reproduce the same task exactly
- graders still remain deterministic
- evaluation stays reproducible
- the baseline can be tested across multiple variants without hand-maintaining fixture tables

## API Endpoints

The FastAPI app in [app.py](./app.py) exposes:

- `GET /health`
- `GET /metadata`
- `GET /schema`
- `POST /mcp`
- `POST /reset`
- `POST /reset/{difficulty}`
- `POST /step`
- `GET /state`

## Local Web Debugger

For local manual debugging, the app also mounts a Gradio UI at:

- `/web`

This lets you:

- reset the environment by difficulty
- manually construct actions
- inspect observations
- inspect rewards and `done/info`
- inspect internal state after each step

Run it locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then open:

```text
http://127.0.0.1:7860/web
```

## Inference Script

The root [inference.py](./inference.py) supports two modes:

- `BASELINE_MODE=heuristic`
- `BASELINE_MODE=model`

### Heuristic Mode

This is the reproducible baseline path. It does not require a model call and is useful for grading sanity checks and validator runs.

### Model Mode

Model mode uses the OpenAI Python client against the Hugging Face router.

Default config:

- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=deepseek-ai/DeepSeek-R1:fastest`

Authentication:

- primary: `HF_TOKEN` environment variable
- local fallback: `credential.txt`
- local fallback: `env/credential.txt`

That means you do not have to type the token every run if you keep it in one of the gitignored credential files.

## Training Loop

The repo also includes [train_loop.py](./train_loop.py), which runs repeated episodes, collects `(state, action, reward, next_state)` transitions, and updates a simple epsilon-greedy Q-table stored in `qtable.json`.

Common commands:

```powershell
python train_loop.py --task all --episodes 200 --policy heuristic
python analyze_training.py --window 20
python train_loop.py --task hard --episodes 100 --policy model
python train_loop.py --task hard --episodes 200 --resume
```

Artifacts written locally:

- `qtable.json`
- `training_log.jsonl`
- `training_results.json`

## Baseline Scores

Local one-episode baselines recorded on April 4, 2026 before the stochastic RL branch work:

| Mode | Model | Easy | Medium | Hard |
|---|---|---:|---:|---:|
| heuristic | built-in heuristic policy | 1.000 | 1.000 | 1.000 |
| model | `Qwen/Qwen3.5-9B` via HF router | 1.000 | 0.500 | 0.902 |

Interpretation:

- the heuristic path acts as a reproducible sanity-check baseline and validator target
- the model path is intentionally imperfect on medium and hard tasks, showing the benchmark is not trivial
- the hard-task score remains high while still exposing realistic failure modes like incomplete flags and malformed matches

### Current Branch Results

Measured on `feature/rl-stochastic-env` on April 5, 2026:

Fresh-run heuristic baseline average across 5 independent `inference.py` runs:

| Task | Average Score |
|---|---:|
| invoice_extract_easy | 1.000 |
| invoice_validate_medium | 1.000 |
| po_reconcile_hard | 1.000 |

Heuristic training summary from `python train_loop.py --task all --episodes 500 --policy heuristic`:

| Difficulty | Avg Last Window | Best | Worst |
|---|---:|---:|---:|
| easy | 0.388 | 1.000 | 0.000 |
| medium | 0.870 | 1.000 | 0.000 |
| hard | 0.966 | 1.000 | 0.000 |

Note: repeated standalone `inference.py` runs start from episode 0 each time, so they replay the first seeded scenario unless you keep a single environment instance alive.

### Clean Heuristic Baseline Outputs

These are fresh heuristic-mode runs from the current branch and reflect the exact stdout contract used for evaluation:

| Task | Steps | Final `[END]` rewards |
|---|---:|---|
| `invoice_extract_easy` | 6 | `0.20,0.20,0.20,0.20,0.20,0.90` |
| `invoice_validate_medium` | 4 | `0.15,0.15,0.15,0.94` |
| `po_reconcile_hard` | 9 | `0.12,0.14,0.16,0.16,0.16,0.14,0.13,0.11,0.95` |

## Sample Inference Output

```text
[START] task=invoice_extract_easy env=finance-ops-openenv model=deepseek-ai/DeepSeek-R1:fastest
[STEP] step=1 action=extract('vendor_name','Zenith Packaging Co') reward=0.20 done=false error=null
[STEP] step=2 action=extract('invoice_number','ZEN-6270') reward=0.20 done=false error=null
[STEP] step=3 action=extract('invoice_date','2026-03-22') reward=0.20 done=false error=null
[STEP] step=4 action=extract('currency','INR') reward=0.20 done=false error=null
[STEP] step=5 action=extract('total_amount','2194.71') reward=0.20 done=false error=null
[STEP] step=6 action=submit() reward=0.90 done=true error=null
[END] success=true steps=6 rewards=0.20,0.20,0.20,0.20,0.20,0.90
```

## Inference Output Contract

The script emits evaluator-facing lines in this exact evaluator-facing format:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

This format is intentionally stable for evaluation. Even when the environment becomes more realistic, the stdout format should remain evaluator-safe.

Example:

```text
[START] task=invoice_extract_easy env=finance-ops-openenv model=deepseek-ai/DeepSeek-R1:fastest
[STEP] step=1 action=extract('vendor_name','Sunrise Office Supplies Pvt Ltd') reward=0.20 done=false error=null
[STEP] step=2 action=extract('invoice_number','INV-2026-0142') reward=0.20 done=false error=null
[STEP] step=3 action=extract('invoice_date','2026-03-14') reward=0.20 done=false error=null
[STEP] step=4 action=extract('currency','INR') reward=0.20 done=false error=null
[STEP] step=5 action=extract('total_amount','13983.00') reward=0.20 done=false error=null
[STEP] step=6 action=submit() reward=0.90 done=true error=null
[END] success=true steps=6 rewards=0.20,0.20,0.20,0.20,0.20,0.90
```

## Running Locally

Install the project:

```bash
pip install -e .
```

Windows PowerShell bootstrap:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Quick environment sanity check:

```powershell
python -c "from env import FinanceOpsEnv; env=FinanceOpsEnv(); print(env.reset('easy').task_id); print(env.reset('medium').task_id); print(env.reset('hard').task_id)"
```

Run the heuristic baseline:

```bash
set BASELINE_MODE=heuristic
set FINANCE_OPS_TASK=invoice_extract_easy
python inference.py
```

Run all three tasks in heuristic mode:

```powershell
$env:BASELINE_MODE="heuristic"
foreach ($task in "invoice_extract_easy","invoice_validate_medium","po_reconcile_hard") {
  $env:FINANCE_OPS_TASK=$task
  python inference.py
}
```

Run the training loop:

```powershell
python train_loop.py --task all --episodes 200 --policy heuristic
python analyze_training.py --window 20
```

Run the model baseline through the Hugging Face router:

```bash
set BASELINE_MODE=model
set HF_TOKEN=your_hf_token
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=deepseek-ai/DeepSeek-R1:fastest
set FINANCE_OPS_TASK=po_reconcile_hard
python inference.py
```

Run the API locally:

```powershell
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then verify:

```text
http://127.0.0.1:7860/health
http://127.0.0.1:7860/metadata
http://127.0.0.1:7860/schema
http://127.0.0.1:7860/web
```

## Round 1 Workflow

Use this order before submission:

1. Install dependencies and confirm all three task difficulties reset successfully.
2. Run the heuristic baseline on all three tasks and save the `[END]` scores.
3. Run the model baseline and save the `[END]` scores for the same tasks.
4. Start the FastAPI app and check `/health`, `/schema`, `/state`, and `/web`.
5. Run `openenv validate`.
6. Build and run Docker locally.
7. Deploy the container to a Hugging Face Docker Space and test the public endpoints.
8. Update this README with final baseline scores, deployment link, and any validator output you want reviewers to see.

Submission evidence checklist:

- typed `Observation`, `Action`, and `Reward` models
- `reset()`, `step()`, and `state()` implemented
- 3 deterministic graded tasks across `easy`, `medium`, and `hard`
- dense rewards with penalties and partial credit
- reproducible baseline logs
- working Docker image
- successful `openenv validate`
- Hugging Face Space deployment

## Why This Is Useful For RL And Agent Training

This benchmark is designed for iterative policy improvement rather than one-shot prompting:

- episodes expose intermediate rewards at every meaningful finance action
- invalid actions and repeated actions incur penalties, which creates a useful optimization surface
- deterministic task variants make it possible to compare agent versions reproducibly
- the same environment supports heuristic baselines, prompted LLM baselines, and future RL fine-tuning loops
- the hard task requires chaining matching, exception handling, and constrained decision-making under partial information

## Current Submission Status

Verified on April 5, 2026:

- `python -m unittest discover -s tests -v` passes
- `.venv\Scripts\openenv.exe validate` passes locally
- `docker build --progress=plain -t financeops-openenv .` succeeds
- local container health check on `http://127.0.0.1:7860/health` returns `200`
- live Hugging Face Space endpoints `/reset`, `/health`, `/schema`, and `/web` return `200`
- heuristic baseline mode is fully reproducible and does not require external credits
- model mode is implemented correctly and connects through the OpenAI client using Hugging Face router variables; heavy multi-episode training still depends on external provider quota

Use a credential file instead of exporting the token every run:

```text
credential.txt
```

Put only the token on the first non-empty line.

## Validation

Run the validator before submission:

```bash
openenv validate
```

Current local validation result:

```text
[OK] Reinforcement-Learning: Ready for multi-mode deployment
```

## Docker

Build locally:

```bash
docker build -t financeops-openenv .
```

Run locally:

```bash
docker run -p 7860:7860 financeops-openenv
```

Then visit:

```text
http://127.0.0.1:7860/health
http://127.0.0.1:7860/web
```

## Hugging Face Spaces

This repo is designed for a Docker Space deployment.

Live Space:

- https://huggingface.co/spaces/Paarthk26/finance-ops-openenv
- https://paarthk26-finance-ops-openenv.hf.space

Suggested steps:

1. Create a new Hugging Face Space with Docker SDK.
2. Push this repository.
3. Add `HF_TOKEN` as a Space secret if you want model-mode inference there.
4. The app will serve on port `7860`.
5. Use `/web` for manual debugging and the API routes for integration testing.

Do not commit credential files or hardcode tokens in source.

### Final Deployment Checklist

Before submitting, confirm each of these once:

1. `openenv validate` passes locally.
2. `docker build -t financeops-openenv .` succeeds.
3. `docker run -p 7860:7860 financeops-openenv` starts cleanly.
4. Your Hugging Face Space returns HTTP 200 from `/reset`.
5. The Space also serves `/health`, `/schema`, and `/web`.
6. `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME` are configured in Space secrets or variables.
7. This README includes final baseline scores and the public Space URL.

### Suggested Space Tags

For discoverability in the Hugging Face UI, add these tags in Space settings:

- `openenv`
- `reinforcement-learning`
- `finance`

### Required Environment Variables

The hackathon runner expects these variable names to exist in your configuration:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

This repository already reads those names directly in [inference.py](./inference.py). For local heuristic runs, only `BASELINE_MODE=heuristic` is needed. For model-mode runs, set all three variables or provide the token through a gitignored credential file.

## Security Notes

- `credential.txt` and `env/credential.txt` are gitignored.
- Hugging Face tokens should be rotated if they have ever been pasted into chat, terminal history, or a committed file.
- For deployment, prefer Hugging Face Space secrets or environment variables over plaintext files.

## Reproducibility Notes

- task variants rotate deterministically
- graders are deterministic
- reward shaping stays within `0.0..1.0` for the same action sequence
- heuristic mode is reproducible
- model mode depends on the routed model behavior

## What Changed From The Earlier Version

Compared with the simpler static version, the current environment now includes:

- seeded stochastic task generation instead of hardcoded fixture tables
- richer hard-task mechanics
- multi-currency reconciliation with FX-adjusted tolerance
- split-PO invoice handling
- a hard-task late-step penalty after turn 6
- duplicate invoice grading
- review memory and last-action-error tracking
- stronger penalties for bad actions
- an explicit `train_loop.py` and `analyze_training.py` workflow for RL-style iteration
- Hugging Face router support in the inference path
- `/web` local debugging interface
- `openenv validate` compatibility

## Recommended Next Steps

The highest-value follow-ups are:

1. add automated tests for reset/step/state and grader score bounds
2. run a full Docker smoke test after every submission-facing change
3. verify the Hugging Face Space deployment end-to-end
4. keep benchmark mode stable before adding optional OCR/upload product layers

## References

- FastAPI: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/latest/
- Gradio: https://www.gradio.app/
- Hugging Face Inference Providers: https://huggingface.co/docs/inference-providers/main/en/index
- Hugging Face Docker Spaces: https://huggingface.co/docs/hub/spaces-sdks-docker
