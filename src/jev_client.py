"""Client for TypeSafe AI's System One API (Jev).

One endpoint, two questions, zero text generation:

    POST {api_url}                      # default https://api.typesafe.ai/v1/systemone
    Authorization: Bearer <TYPESAFE_API_KEY>

    {
      "model": "jev-1.13.0",
      "state": {"query": "<user text>"},
      "questions": {
        "intent": {
          "type": "choice",
          "instructions": "Which intent best matches the user's query?",
          "criteria": {"billing_inquiry": "<description>", ...}
        },
        "human_review": {
          "type": "noul",
          "instructions": "Is this query risky, ambiguous, or out of scope enough
                           that a human should review it before any worker acts?"
        }
      }
    }

Response (answers echoed by our question ids):

    {
      "model": "jev-1.13.0",
      "answers": {
        "intent": {"type": "choice", "choice": "billing_inquiry",
                   "probabilities": {...}, "confidence": 0.91},
        "human_review": {"type": "noul", "noul": 0.12}
      },
      "usage": {"input_tokens": 180, "output_tokens": 0}
    }

Uses raw httpx — the official ``typesafe-sdk`` is optional and not required.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

from .intents import Intent

log = logging.getLogger(__name__)

#: HTTP statuses that are worth retrying with exponential backoff.
#: 429 = rate limited (250k tokens/s or 1,200 req/min), 529 = provider overloaded.
RETRY_STATUSES = {408, 429, 529}


class JevError(RuntimeError):
    """Raised when the System One API call fails unrecoverably."""


@dataclass
class JevDecision:
    intent: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    needs_human: float = 0.0
    model: str = ""
    latency_ms: float = 0.0
    usage: dict = field(default_factory=dict)


def build_questions(intents: list[Intent]) -> dict:
    """Build the two-question payload from the intent taxonomy.

    The intent *descriptions* become the Choice criteria, so adding an
    intent to data/intents.yaml is all it takes to teach Jev a new
    category — no fine-tuning, no prompt surgery.
    """
    criteria = {i.name: i.description for i in intents}
    return {
        "intent": {
            "type": "choice",
            "instructions": (
                "Which intent best matches the user's query? Pick the single "
                "option whose description fits best. If none fits clearly, "
                "pick the fallback option."
            ),
            "criteria": criteria,
        },
        "human_review": {
            "type": "noul",
            "instructions": (
                "Is this query risky, ambiguous, or out of scope enough that a "
                "human should review it before any worker agent acts on it? "
                "Answer with the probability that yes, a human is needed."
            ),
        },
    }


def parse_decision(
    payload: dict, intents: list[Intent], *, model: str = "", latency_ms: float = 0.0
) -> JevDecision:
    """Parse a System One response into a JevDecision, defensively."""
    allowed = {i.name for i in intents}
    answers = (payload or {}).get("answers", {})

    intent_answer = answers.get("intent", {})
    choice = str(intent_answer.get("choice", "fallback"))
    if choice not in allowed:
        log.warning("Jev returned unknown intent %r; falling back", choice)
        choice = "fallback"
    probabilities = {
        str(k): float(v)
        for k, v in (intent_answer.get("probabilities") or {}).items()
        if str(k) in allowed
    }
    try:
        confidence = float(intent_answer.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    human_answer = answers.get("human_review", {})
    try:
        needs_human = float(human_answer.get("noul", 0.0))
    except (TypeError, ValueError):
        needs_human = 0.0
    needs_human = max(0.0, min(1.0, needs_human))

    return JevDecision(
        intent=choice,
        probabilities=probabilities,
        confidence=confidence,
        needs_human=needs_human,
        model=str((payload or {}).get("model", model)),
        latency_ms=latency_ms,
        usage=dict((payload or {}).get("usage", {})),
    )


class JevClient:
    """Thin wrapper around the System One HTTP API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "jev-1.13.0",
        api_url: str = "https://api.typesafe.ai/v1/systemone",
        timeout_s: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for the Jev backend")
        self.api_key = api_key
        self.model = model
        self.api_url = api_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def decide(self, text: str, intents: list[Intent]) -> JevDecision:
        """Ask Jev: which intent, and does a human need to look?"""
        body = {
            "model": self.model,
            "state": {"query": text},
            "questions": build_questions(intents),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            try:
                resp = httpx.post(
                    self.api_url, json=body, headers=headers, timeout=self.timeout_s
                )
            except Exception as exc:
                last_exc = exc
                log.warning("Jev request failed (attempt %d): %s", attempt + 1, exc)
            else:
                latency_ms = (time.monotonic() - started) * 1000.0
                if resp.status_code == 200:
                    return parse_decision(
                        resp.json(), intents, model=self.model, latency_ms=latency_ms
                    )
                if resp.status_code in RETRY_STATUSES:
                    log.warning(
                        "Jev returned %d (attempt %d); backing off",
                        resp.status_code,
                        attempt + 1,
                    )
                    last_exc = JevError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                elif resp.status_code == 401:
                    raise JevError("HTTP 401: invalid TYPESAFE_API_KEY")
                elif resp.status_code == 422:
                    raise JevError(f"HTTP 422: malformed question: {resp.text[:200]}")
                else:
                    raise JevError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            # exponential backoff: 0.5s, 1s, 2s ...
            time.sleep(0.5 * (2**attempt))

        raise JevError(f"Jev request failed after {self.max_retries + 1} attempts: {last_exc}")
