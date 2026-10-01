"""Runtime configuration for the intent-router-jeV service.

Every setting comes from an environment variable so the same container image
can run locally, in docker-compose, and on GKE via Helm with no code changes.
"""
from __future__ import annotations

import json
import os


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Settings:
    """All runtime configuration, read from the environment."""

    #: Default A2A worker URLs used when WORKER_AGENTS_JSON is not set.
    #: These match the docker-compose demo (worker-a on :8001, worker-b on :8002).
    DEFAULT_WORKERS: dict[str, str] = {
        "billing_inquiry": "http://worker-a:8001",
        "technical_support": "http://worker-a:8001",
        "sales_question": "http://worker-a:8001",
        "account_update": "http://worker-b:8002",
        "order_status": "http://worker-b:8002",
        "fallback": "http://worker-b:8002",
    }

    def __init__(self) -> None:
        backend = _get("CLASSIFIER_BACKEND", "mock").strip().lower()
        if backend not in ("mock", "jev"):
            raise ValueError(
                f"CLASSIFIER_BACKEND must be one of mock|jev, got {backend!r}"
            )
        self.classifier_backend: str = backend

        self.typesafe_api_key: str = _get("TYPESAFE_API_KEY", "")
        self.typesafe_model: str = _get("TYPESAFE_MODEL", "jev-1.13.0")
        self.typesafe_api_url: str = _get(
            "TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone"
        ).rstrip("/")
        try:
            self.typesafe_timeout_s: float = float(_get("TYPESAFE_TIMEOUT_S", "10"))
        except ValueError as exc:
            raise ValueError("TYPESAFE_TIMEOUT_S must be a number") from exc

        if backend == "jev" and not self.typesafe_api_key:
            raise ValueError(
                "TYPESAFE_API_KEY must be set when CLASSIFIER_BACKEND=jev"
            )

        try:
            self.confidence_threshold: float = float(_get("CONFIDENCE_THRESHOLD", "0.6"))
        except ValueError as exc:
            raise ValueError("CONFIDENCE_THRESHOLD must be a number") from exc
        try:
            self.human_review_threshold: float = float(
                _get("HUMAN_REVIEW_THRESHOLD", "0.5")
            )
        except ValueError as exc:
            raise ValueError("HUMAN_REVIEW_THRESHOLD must be a number") from exc

        self.worker_agents: dict[str, str] = self._load_worker_agents()
        self.log_level: str = _get("LOG_LEVEL", "INFO").upper()

    @classmethod
    def _load_worker_agents(cls) -> dict[str, str]:
        raw = os.environ.get("WORKER_AGENTS_JSON", "").strip()
        if not raw:
            return dict(cls.DEFAULT_WORKERS)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"WORKER_AGENTS_JSON is not valid JSON: {exc}") from exc
        if not isinstance(data, dict) or not data:
            raise ValueError(
                "WORKER_AGENTS_JSON must be a non-empty JSON object mapping intent -> agent URL"
            )
        return {str(k): str(v) for k, v in data.items()}

    def worker_url(self, intent: str) -> str:
        """Resolve the A2A agent URL for an intent, falling back gracefully."""
        if intent in self.worker_agents:
            return self.worker_agents[intent]
        if "fallback" in self.worker_agents:
            return self.worker_agents["fallback"]
        return next(iter(self.worker_agents.values()))
