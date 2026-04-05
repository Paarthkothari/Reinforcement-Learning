from __future__ import annotations

import json

import gradio as gr
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env import Action, FinanceOpsEnv, Observation

app = FastAPI(title="FinanceOps OpenEnv", version="1.0.0")
env = FinanceOpsEnv()
env.reset("easy")


class ResetRequest(BaseModel):
    difficulty: str = "easy"


@app.get("/", include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>FinanceOps OpenEnv</title>
          <style>
            :root {
              color-scheme: light dark;
              --bg: #0b1320;
              --panel: rgba(15, 23, 42, 0.88);
              --text: #e5eefc;
              --muted: #a9bbd8;
              --accent: #7dd3fc;
              --accent-2: #86efac;
              --border: rgba(148, 163, 184, 0.28);
            }
            * { box-sizing: border-box; }
            body {
              margin: 0;
              min-height: 100vh;
              font-family: Arial, Helvetica, sans-serif;
              background:
                radial-gradient(circle at top left, rgba(34, 197, 94, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.2), transparent 30%),
                linear-gradient(180deg, #020617 0%, #0f172a 100%);
              color: var(--text);
              display: grid;
              place-items: center;
              padding: 24px;
            }
            .card {
              width: min(860px, 100%);
              border: 1px solid var(--border);
              background: var(--panel);
              backdrop-filter: blur(10px);
              border-radius: 20px;
              padding: 28px;
              box-shadow: 0 24px 80px rgba(2, 6, 23, 0.45);
            }
            h1 {
              margin: 0 0 12px;
              font-size: clamp(2rem, 4vw, 3rem);
              line-height: 1.05;
            }
            p {
              margin: 0 0 18px;
              color: var(--muted);
              line-height: 1.6;
            }
            .actions {
              display: flex;
              flex-wrap: wrap;
              gap: 12px;
              margin: 24px 0;
            }
            .button {
              display: inline-flex;
              align-items: center;
              justify-content: center;
              min-width: 180px;
              padding: 12px 18px;
              border-radius: 999px;
              border: 1px solid transparent;
              text-decoration: none;
              font-weight: 700;
            }
            .primary {
              background: linear-gradient(135deg, var(--accent), var(--accent-2));
              color: #082f49;
            }
            .secondary {
              border-color: var(--border);
              color: var(--text);
              background: rgba(15, 23, 42, 0.45);
            }
            ul {
              margin: 20px 0 0;
              padding-left: 18px;
              color: var(--muted);
            }
            li { margin: 8px 0; }
            code {
              color: var(--text);
              background: rgba(148, 163, 184, 0.16);
              padding: 2px 6px;
              border-radius: 6px;
            }
          </style>
        </head>
        <body>
          <main class="card">
            <h1>FinanceOps OpenEnv</h1>
            <p>
              Finance operations benchmark environment for invoice extraction, invoice validation,
              and PO reconciliation. The live debugger is available below.
            </p>
            <div class="actions">
              <a class="button primary" href="/web/">Open Debugger</a>
              <a class="button secondary" href="/health">Health Check</a>
              <a class="button secondary" href="/schema">Schema</a>
              <a class="button secondary" href="/docs">API Docs</a>
            </div>
            <ul>
              <li><code>POST /reset</code> returns a fresh observation.</li>
              <li><code>POST /step</code> advances the environment with a typed action.</li>
              <li><code>/web/</code> hosts the manual Gradio debugger UI.</li>
            </ul>
          </main>
        </body>
        </html>
        """
    )


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "name": "finance-ops-openenv",
        "description": (
            "OpenEnv-compatible finance operations environment for invoice extraction, "
            "validation, and PO reconciliation."
        ),
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {"type": "object"},
    }


@app.post("/mcp")
def mcp() -> dict:
    return {"jsonrpc": "2.0", "result": {"status": "ok"}, "id": None}


@app.post("/reset")
def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> dict:
    try:
        observation = env.reset(request.difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return observation.model_dump()


@app.post("/reset/{difficulty}")
def reset_for_difficulty(difficulty: str) -> dict:
    try:
        observation = env.reset(difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return observation.model_dump()


@app.post("/step")
def step(action: Action) -> dict:
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict:
    return env.state()


def _to_json(payload: dict) -> str:
    return json.dumps(payload, indent=2)


def _reset_ui(difficulty: str) -> tuple[str, str]:
    observation = env.reset(difficulty)
    return _to_json(observation.model_dump()), _to_json(env.state())


def _step_ui(
    action_type: str,
    field_name: str,
    field_value: str,
    issue_code: str,
    invoice_id: str,
    po_id: str,
    notes: str,
) -> tuple[str, str, str, str]:
    action = Action(
        action_type=action_type,
        field_name=field_name or None,
        field_value=field_value or None,
        issue_code=issue_code or None,
        invoice_id=invoice_id or None,
        po_id=po_id or None,
        notes=notes or None,
    )
    observation, reward, done, info = env.step(action)
    return (
        _to_json(observation.model_dump()),
        _to_json(reward.model_dump()),
        _to_json({"done": done, "info": info}),
        _to_json(env.state()),
    )


with gr.Blocks(title="FinanceOps OpenEnv Debugger") as web_demo:
    gr.Markdown("## FinanceOps OpenEnv Debugger")
    gr.Markdown("Use this for local debugging of `reset()` and `step()` at `/web`.")

    with gr.Row():
        difficulty = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="easy",
            label="Difficulty",
        )
        reset_btn = gr.Button("Reset", variant="primary")
        refresh_state_btn = gr.Button("Refresh State")

    observation_out = gr.Textbox(label="Observation", lines=18)
    state_out = gr.Textbox(label="State", lines=18)

    with gr.Row():
        action_type = gr.Dropdown(
            choices=["extract", "flag", "match", "submit", "skip"],
            value="extract",
            label="Action Type",
        )
        field_name = gr.Textbox(label="Field Name")
        field_value = gr.Textbox(label="Field Value")

    with gr.Row():
        issue_code = gr.Textbox(label="Issue Code")
        invoice_id = gr.Textbox(label="Invoice ID")
        po_id = gr.Textbox(label="PO ID")

    notes = gr.Textbox(label="Notes", lines=2)
    step_btn = gr.Button("Step", variant="primary")

    reward_out = gr.Textbox(label="Reward", lines=6)
    info_out = gr.Textbox(label="Done / Info", lines=8)

    reset_btn.click(fn=_reset_ui, inputs=difficulty, outputs=[observation_out, state_out])
    refresh_state_btn.click(fn=lambda: _to_json(env.state()), outputs=state_out)
    step_btn.click(
        fn=_step_ui,
        inputs=[action_type, field_name, field_value, issue_code, invoice_id, po_id, notes],
        outputs=[observation_out, reward_out, info_out, state_out],
    )


app = gr.mount_gradio_app(app, web_demo, path="/web")
