from __future__ import annotations

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
