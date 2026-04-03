# FinanceOps OpenEnv

FinanceOps OpenEnv is an OpenEnv-compatible benchmark environment for finance operations workflows. It models three practical back-office tasks that human teams actually perform:

1. invoice field extraction
2. invoice anomaly validation
3. purchase-order reconciliation

The project is benchmark-first. It is designed for agent training and evaluation, with typed models, deterministic graders, dense rewards, local debugging tools, and deployment support for Docker and Hugging Face Spaces.

## Why This Environment Matters

Finance operations teams routinely process invoices, validate faulty documents, and reconcile vendor invoices against approved purchase orders. These are repetitive, rules-heavy, auditable tasks that are valuable for benchmarking agent reliability.

This environment aims to score well on the typical OpenEnv judging dimensions:

- real-world utility: the task domain is practical and non-toy
- task quality: three tasks with meaningful difficulty progression
- environment design: stateful multi-step interaction with penalties and partial credit
- code quality: typed models, structured repo, Docker support, validator support
- creativity: richer hard-task mechanics like vendor aliases, duplicate invoices, split invoices, and bad PO references

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

Realism improvements:
- vendor aliases
- PO references
- conflicting vendor names
- split invoices / partial payments
- wrong PO references
- duplicate invoice submissions
- amount tolerance rules

Grader:
- match accuracy
- unmatched detection
- amount mismatch detection
- duplicate invoice detection

All graders are deterministic and return values in `0.0..1.0`.

## Environment Design

The environment uses dense rewards rather than only binary success/failure.

Positive signals:
- correct extraction
- correct anomaly flag
- correct PO match
- correct discrepancy flag
- correct duplicate-invoice flag

Negative signals:
- wrong extraction
- wrong match
- false positive flags
- repeated flags or repeated matches
- invalid actions
- skipping turns
- incomplete submission through readiness penalties

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

Task payloads are not single hardcoded examples anymore. Variants are defined in [env/data.py](./env/data.py), and `reset()` cycles through them deterministically per difficulty.

That means:

- repeated resets expose different realistic cases
- graders still remain deterministic
- evaluation stays reproducible
- the baseline can be tested across multiple variants

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

## Inference Output Contract

The script emits evaluator-facing lines in this format:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
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
[END] success=true steps=6 score=1.000 rewards=0.20,0.20,0.20,0.20,0.20,0.90
```

## Running Locally

Install the project:

```bash
pip install -e .
```

Run the heuristic baseline:

```bash
set BASELINE_MODE=heuristic
set FINANCE_OPS_TASK=invoice_extract_easy
python inference.py
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
[OK] Reinforcement: Ready for multi-mode deployment
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

Suggested steps:

1. Create a new Hugging Face Space with Docker SDK.
2. Push this repository.
3. Add `HF_TOKEN` as a Space secret if you want model-mode inference there.
4. The app will serve on port `7860`.
5. Use `/web` for manual debugging and the API routes for integration testing.

Do not commit credential files or hardcode tokens in source.

## Security Notes

- `credential.txt` and `env/credential.txt` are gitignored.
- Hugging Face tokens should be rotated if they have ever been pasted into chat, terminal history, or a committed file.
- For deployment, prefer Hugging Face Space secrets or environment variables over plaintext files.

## Reproducibility Notes

- task variants rotate deterministically
- graders are deterministic
- reward shaping is deterministic for the same action sequence
- heuristic mode is reproducible
- model mode depends on the routed model behavior

## What Changed From The Earlier Version

Compared with the simpler static version, the current environment now includes:

- deterministic generated task variants
- richer hard-task mechanics
- duplicate invoice grading
- review memory and last-action-error tracking
- stronger penalties for bad actions
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
