"""Router agent: classify -> dual gate -> A2A dispatch.

The router is intentionally thin: the classifier decides the intent, two
gates decide whether we trust it, and the A2A client delivers the query to
the worker agent that owns the intent.

Gates (both must pass for automatic routing):
  1. confidence gate — Choice confidence below CONFIDENCE_THRESHOLD
  2. human gate     — Noul "needs human" probability above HUMAN_REVIEW_THRESHOLD

Either gate failing sends the query to the ``fallback`` worker (your human
queue), and the response says which gate tripped so you can tune thresholds
from data.
"""
from __future__ import annotations

import logging
import time

from .a2a_client import A2AClient
from .classifier import IntentClassifier

log = logging.getLogger(__name__)


class RouterAgent:
    def __init__(
        self,
        classifier: IntentClassifier,
        worker_agents: dict[str, str],
        confidence_threshold: float = 0.6,
        human_review_threshold: float = 0.5,
        a2a_client: A2AClient | None = None,
    ) -> None:
        if not worker_agents:
            raise ValueError("worker_agents must not be empty")
        self.classifier = classifier
        self.worker_agents = worker_agents
        self.confidence_threshold = confidence_threshold
        self.human_review_threshold = human_review_threshold
        self.a2a = a2a_client or A2AClient()

    def resolve_worker(self, intent: str) -> str:
        if intent in self.worker_agents:
            return self.worker_agents[intent]
        if "fallback" in self.worker_agents:
            return self.worker_agents["fallback"]
        return next(iter(self.worker_agents.values()))

    def handle_query(self, text: str) -> dict:
        """Classify a query and dispatch it to the owning worker agent."""
        classification = self.classifier.classify(text)
        intent = classification.intent
        gates: dict[str, bool] = {"low_confidence": False, "human_review": False}

        if classification.confidence < self.confidence_threshold:
            log.info(
                "confidence %.3f below threshold %.3f -> routing to fallback",
                classification.confidence,
                self.confidence_threshold,
            )
            gates["low_confidence"] = True
        if classification.needs_human > self.human_review_threshold:
            log.info(
                "needs_human %.3f above threshold %.3f -> routing to fallback",
                classification.needs_human,
                self.human_review_threshold,
            )
            gates["human_review"] = True

        if any(gates.values()):
            intent = "fallback"

        worker_url = self.resolve_worker(intent)
        dispatch = self.a2a.dispatch(worker_url, text, intent)
        return {
            "intent": intent,
            "confidence": round(classification.confidence, 3),
            "probabilities": {
                k: round(v, 3) for k, v in classification.probabilities.items()
            },
            "needs_human": round(classification.needs_human, 3),
            "gates": gates,
            "routed_to_fallback": any(gates.values()),
            "worker": worker_url,
            "task_id": dispatch["task_id"],
            "dispatch": dispatch,
            "ts": time.time(),
        }
