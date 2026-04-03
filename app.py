from __future__ import annotations

import json

import gradio as gr
from fastapi import FastAPI, HTTPException

from env import Action, FinanceOpsEnv, Observation

app = FastAPI(title="FinanceOps OpenEnv", version="1.0.0")
env = FinanceOpsEnv()
env.reset("easy")


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
def reset() -> dict:
    observation = env.reset("easy")
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
