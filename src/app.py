"""FastAPI service exposing the Jev intent router.

Endpoints:
  GET  /health    - liveness/readiness probe
  POST /classify  - {"text": "..."} -> {"intent", "confidence",
                                        "probabilities", "needs_human"}
  POST /route     - {"text": "..."} -> full routing decision + A2A dispatch
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .a2a_client import A2AClient
from .classifier import IntentClassifier
from .config import Settings
from .intents import intent_names, load_intents
from .router import RouterAgent

log = logging.getLogger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    intents = load_intents()
    classifier = IntentClassifier(
        intents,
        backend=settings.classifier_backend,
        jev_api_key=settings.typesafe_api_key,
        jev_model=settings.typesafe_model,
        jev_api_url=settings.typesafe_api_url,
        jev_timeout_s=settings.typesafe_timeout_s,
    )
    router = RouterAgent(
        classifier,
        settings.worker_agents,
        confidence_threshold=settings.confidence_threshold,
        human_review_threshold=settings.human_review_threshold,
        a2a_client=A2AClient(),
    )
    _state.update(settings=settings, intents=intents, router=router)
    log.info(
        "intent-router-jev ready: backend=%s model=%s intents=%s conf=%.2f human=%.2f",
        settings.classifier_backend,
        settings.typesafe_model if settings.classifier_backend == "jev" else "-",
        intent_names(intents),
        settings.confidence_threshold,
        settings.human_review_threshold,
    )
    yield
    _state.clear()


app = FastAPI(title="Intent Router (Jev)", version="0.2.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class ClassificationResponse(BaseModel):
    intent: str
    confidence: float
    probabilities: dict[str, float] = {}
    needs_human: float = 0.0


def _router() -> RouterAgent:
    router = _state.get("router")
    if router is None:
        raise HTTPException(status_code=503, detail="router not initialized")
    return router


@app.get("/health")
def health():
    intents = _state.get("intents", [])
    settings = _state.get("settings")
    return {
        "status": "ok",
        "backend": settings.classifier_backend if settings else None,
        "model": settings.typesafe_model if settings else None,
        "intents": intent_names(intents),
    }


@app.post("/classify", response_model=ClassificationResponse)
def classify(req: QueryRequest):
    c = _router().classifier.classify(req.text)
    return ClassificationResponse(
        intent=c.intent,
        confidence=c.confidence,
        probabilities=c.probabilities,
        needs_human=c.needs_human,
    )


@app.post("/route")
def route(req: QueryRequest):
    return _router().handle_query(req.text)
