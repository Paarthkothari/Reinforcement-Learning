# FinanceOps OpenEnv

FinanceOps OpenEnv is a real-world training environment for AI agents working on SMB finance operations. It simulates three common workflows:

1. Invoice field extraction
2. Invoice anomaly validation
3. Purchase-order reconciliation

The environment exposes the standard `reset()`, `step()`, and `state()` interface with typed Pydantic models and a FastAPI wrapper for local testing or deployment to Hugging Face Spaces.

## Why this environment is useful

This domain is practical rather than game-like. Finance teams routinely extract invoice fields, validate faulty invoices, and reconcile invoices against approved purchase orders. Those workflows are repetitive, rules-driven, and measurable, which makes them a strong fit for agent evaluation.

## Project structure

```text
.
|-- app.py
|-- Dockerfile
|-- env
|   |-- __init__.py
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
|-- README.md
`-- requirements.txt
```

## Environment API

### Observation

The observation is what the agent sees on each step.

```json
{
  "task_id": "invoice_extract_easy",
  "difficulty": "easy",
  "content": {
    "instructions": "Extract the required fields from the invoice text, then submit when you are confident.",
    "document": {
      "invoice_text": "Vendor: ...",
      "required_fields": ["vendor_name", "invoice_number", "invoice_date", "currency", "total_amount"]
    },
    "current_submission": {}
  },
  "available_actions": ["extract", "flag", "match", "submit", "skip"],
  "step_number": 0,
  "max_steps": 8,
  "context": "Finance operations task: extraction"
}
```

### Action

The action is what the agent sends back to the environment.

```json
{
  "action_type": "extract",
  "field_name": "invoice_number",
  "field_value": "INV-2026-0142"
}
```

Supported action types:

- `extract`: used in the easy extraction task
- `flag`: used in validation and reconciliation
- `match`: used in reconciliation
- `submit`: ends the episode and triggers final grading
- `skip`: no-op with a small penalty

### Reward

Each step returns a typed reward with dense signal:

```json
{
  "score": 0.2,
  "reason": "Correct extraction for 'invoice_number'.",
  "partial_credit": 0.2
}
```

Reward shaping rules:

- Correct field extraction: positive reward
- Partial extraction: smaller positive reward
- Correct anomaly flags: positive reward
- False positives and wrong matches: negative reward
- Late submissions: step penalty
- Repeated or irrelevant actions: small penalty

## Tasks and graders

### Easy: invoice extraction

- Goal: extract `vendor_name`, `invoice_number`, `invoice_date`, `currency`, `total_amount`
- Grader: exact normalized field match
- Output score: `correct_fields / total_fields`

### Medium: invoice validation

- Goal: detect all real anomalies in a faulty invoice
- Ground-truth issues:
  - `invalid_invoice_date`
  - `duplicate_line_item`
  - `subtotal_mismatch`
  - `missing_gstin`
- Grader: recall with false-positive penalty

### Hard: PO reconciliation

- Goal: match invoices to purchase orders and flag unmatched invoices
- Grader:
  - 75% invoice-to-PO match accuracy
  - 25% unmatched invoice detection with false-positive penalty

## Local setup

Create a virtual environment, install dependencies, and run the API:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

API endpoints:

- `GET /health`
- `POST /reset`
- `POST /reset/{difficulty}`
- `POST /step`
- `GET /state`

## Minimal usage example

```python
from env import Action, FinanceOpsEnv

env = FinanceOpsEnv()
obs = env.reset("easy")
print(obs.model_dump())

obs, reward, done, info = env.step(
    Action(action_type="extract", field_name="invoice_number", field_value="INV-2026-0142")
)
print(reward.model_dump())
```

## Inference script

The root `inference.py` uses the OpenAI Python client, but it now defaults to Groq's OpenAI-compatible API. It reads:

- `GROQ_API_KEY` preferred
- `OPENAI_API_KEY` or `HF_TOKEN` as fallback
- `API_BASE_URL` optional, defaults to `https://api.groq.com/openai/v1`
- `MODEL_NAME` optional, defaults to `llama-3.1-8b-instant`

Run it like this:

```bash
set GROQ_API_KEY=your_key
set MODEL_NAME=llama-3.1-8b-instant
python inference.py
```

If you want to be explicit, you can also set:

```bash
set API_BASE_URL=https://api.groq.com/openai/v1
```

Expected output format:

```text
easy: total_reward=0.7600 task_score=1.0000 steps=6
medium: total_reward=0.4600 task_score=0.7500 steps=5
hard: total_reward=0.3100 task_score=0.8333 steps=6
mean_task_score=0.8611
```

These sample numbers are illustrative. Your exact baseline depends on the Groq model behind `MODEL_NAME`.

## Docker

Build and run locally:

```bash
docker build -t financeops-openenv .
docker run -p 7860:7860 financeops-openenv
```

## Hugging Face Spaces

This repo is ready for a Docker Space:

1. Create a new Hugging Face Space with Docker SDK.
2. Push this repository.
3. Ensure the Space has the `openenv` tag in its metadata or description.
4. The app will serve on port `7860`.

## Suggested learning plan

If you want to understand every part before extending it, follow this order:

1. Python classes and dictionaries
2. Pydantic models
3. FastAPI routing
4. Environment loop design (`reset`, `step`, `state`)
5. Docker basics
6. Model-driven inference with the OpenAI client

Helpful references:

- Pydantic: https://docs.pydantic.dev/latest/
- FastAPI: https://fastapi.tiangolo.com/
- OpenAI API: https://platform.openai.com/docs/api-reference
- Docker getting started: https://docs.docker.com/get-started/
- Hugging Face Docker Spaces: https://huggingface.co/docs/hub/spaces-sdks-docker

## Notes on reproducibility

- Task data is deterministic and embedded locally.
- Graders are deterministic and return scores in the `0.0` to `1.0` range.
- Rewards are shaped but deterministic for identical action sequences.
- Inference reproducibility improves by keeping temperature very low.

## Groq compatibility notes

This project uses Groq through the official OpenAI compatibility layer. According to Groq's official docs, the OpenAI client should be pointed at `https://api.groq.com/openai/v1` and authenticated with `GROQ_API_KEY`.

Sources:

- https://console.groq.com/docs/openai
- https://console.groq.com/docs/models
