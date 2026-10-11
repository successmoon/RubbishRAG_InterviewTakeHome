"""RubbishRAG minimal API server (FastAPI + auto Swagger).

Run:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
Swagger UI:
    http://localhost:8000/docs        (interactive test page)
    http://localhost:8000/redoc       (reference docs)
    http://localhost:8000/openapi.json (raw schema)

Endpoints:
    GET  /health          service + index stats
    POST /retrieve        flawed hybrid retrieval (the black box)
    GET  /visible         visible bench queries (no gold answers)
    POST /submit          score predictions vs HIDDEN set (aggregates only, no leak)
    GET  /quota/{api_key} quota usage
"""
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="RubbishRAG Black-Box API",
    description=("Flawed hybrid retriever for the RubbishRAG interview take-home. "
                 "AI made it in 1 minute — prove it wrong. "
                 "Use POST /retrieve to probe it, POST /submit to get scored."),
    version="1.0.0",
)


# ---------------------------------------------------------------- models

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Persian query, e.g. 'رتبه C1 یعنی چی؟'",
                       examples=["رتبه C1 یعنی چی؟"])
    topk: int = Field(5, ge=1, le=10, description="How many chunks to return")
    api_key: str = Field("demo-key", description="Per-candidate key (quota + jitter)")


class Hit(BaseModel):
    doc_id: int
    chunk_id: str
    text: str
    category: str = ""
    bm25: float
    dense: float
    fused: float


class RetrieveResponse(BaseModel):
    hits: list[Hit]
    quota_used: int


class Prediction(BaseModel):
    id: str = Field(..., description="Bench item id, e.g. 'h-base-0'")
    status: str = Field(..., description="'answer' or 'clarify'",
                        examples=["answer"])
    cites: list[int] = Field(default_factory=list,
                             description="Top doc_ids (gold must be in first 3)")
    brief: str = Field("", description="1-line answer (should carry numbers/ranks)")


class SubmitRequest(BaseModel):
    api_key: str = Field("demo-key", description="Candidate key (logged)")
    predictions: list[Prediction] = Field(...,
        description="One prediction per hidden item id (get ids from local "
                    "visible set for practice; hidden ids are h-base-*/h-swap-*/h-amb-*/h-ood-*)")


class SubmitResponse(BaseModel):
    total: int
    accuracy: float
    answer_acc: float
    clarify_acc: float
    n_scored: int


# ---------------------------------------------------------------- helpers

def _quota_used(api_key: str) -> int:
    p = os.path.join(BASE, "traces", "quota.json")
    try:
        return int(json.load(open(p, encoding="utf-8")).get(api_key, 0))
    except Exception:
        return 0


def _load_hidden():
    from .hidden_eval_builder import load_hidden
    return load_hidden()


# ---------------------------------------------------------------- routes

@app.get("/health", summary="Service + index stats")
def health():
    from . import _hidden_retriever as hr
    idx = hr._load_index()
    return {"ok": True, "chunks": idx["N"],
            "avgdl": round(idx["avgdl"], 1),
            "note": "RubbishRAG black box is up. Prove it wrong."}


@app.post("/retrieve", response_model=RetrieveResponse,
          summary="Flawed hybrid retrieval (the black box)")
def retrieve(req: RetrieveRequest):
    from . import _hidden_retriever as hr
    try:
        hits = hr.remote_retrieve(req.query, topk=req.topk, api_key=req.api_key)
    except RuntimeError:
        raise HTTPException(status_code=429, detail="quota exceeded (max 1000)")
    return {"hits": hits, "quota_used": _quota_used(req.api_key)}


@app.get("/visible", summary="Visible bench queries (practice, no gold)")
def visible():
    p = os.path.join(BASE, "server", "visible_bench.json")
    if not os.path.exists(p):
        from .hidden_eval_builder import build
        build()
    items = json.load(open(p, encoding="utf-8"))
    return [{"id": it["id"], "query": it["query"]} for it in items]


@app.post("/submit", response_model=SubmitResponse,
          summary="Score predictions vs HIDDEN set (aggregates only)")
def submit(req: SubmitRequest):
    from .evaluator import score_one, aggregate
    hidden = {h["id"]: h for h in _load_hidden()}
    results = []
    for pred in req.predictions:
        gold = hidden.get(pred.id)
        if gold is None:
            continue
        results.append(score_one(pred.model_dump(), gold))
    if not results:
        raise HTTPException(status_code=400,
                            detail="no matching hidden ids in predictions")
    agg = aggregate(results)
    # log submission (observability per candidate key)
    try:
        os.makedirs(os.path.join(BASE, "traces"), exist_ok=True)
        with open(os.path.join(BASE, "traces", "submissions.jsonl"),
                  "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": req.api_key, **agg},
                               ensure_ascii=False) + "\n")
    except Exception:
        pass
    return {**agg, "n_scored": len(results)}


@app.get("/quota/{api_key}", summary="Quota usage for a key")
def quota(api_key: str):
    from ._secrets import QUOTA
    return {"key": api_key, "used": _quota_used(api_key),
            "max": QUOTA["max_retrieve"]}


class StudioQueryRequest(BaseModel):
    query: str
    mode: str = "orchestrator"  # "naive" or "orchestrator"
    topk: int = 5


@app.get("/", response_class=HTMLResponse, summary="RubbishRAG Interactive Studio UI")
@app.get("/ui", response_class=HTMLResponse, summary="RubbishRAG Interactive Studio UI")
def serve_ui():
    ui_path = os.path.join(BASE, "server", "ui.html")
    if os.path.exists(ui_path):
        with open(ui_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Studio UI template not found</h2>", status_code=404)


@app.post("/studio/query", summary="Interactive studio query endpoint")
def studio_query(req: StudioQueryRequest):
    if req.mode == "orchestrator":
        try:
            from rubbish_rag.orchestrator import get_orchestrator
            orch = get_orchestrator()
            res = orch.run(req.query, topk=req.topk)
            # Ensure hits format
            if "hits" not in res and res.get("status") == "clarify":
                # provide hits for inspection
                from server import _hidden_retriever as hr
                raw_hits = hr.remote_retrieve(req.query, topk=req.topk, api_key="studio")
                res["hits"] = raw_hits
            return res
        except Exception as e:
            return {"status": "error", "brief": f"Orchestrator error: {str(e)}", "cites": [], "hits": []}
    else:
        # Naive pipeline
        import rubbish_rag as R
        import rubbish_rag.pipeline  # noqa
        from server import _hidden_retriever as hr
        hits = hr.remote_retrieve(req.query, topk=req.topk, api_key="studio")
        reranked = R.get("reranker")(req.query, hits)
        pred = R.get("resolver")(req.query, reranked)
        return {
            "status": pred.get("status", "answer"),
            "brief": pred.get("brief", ""),
            "cites": pred.get("cites", []),
            "hits": reranked
        }

