"""Intent classification with pluggable backends.

Backends:
  * ``mock`` - deterministic keyword scoring shaped like a Jev response
               (probabilities + confidence + needs_human). Zero dependencies.
               Used for local dev, CI, and wiring tests.
  * ``jev``  - live TypeSafe System One API via :class:`JevClient`
               (JevClient.decide). Requires TYPESAFE_API_KEY.

Unlike the LLM-based router, there is no prompt to build and no text to
parse: Jev's output space is closed by the questions we send, so the answer
is directly consumable by code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .intents import Intent
from .jev_client import JevClient, JevError

log = logging.getLogger(__name__)

#: Keyword lists for the mock backend. Keep these aligned with the example
#: utterances in data/intents.yaml.
_MOCK_KEYWORDS: dict[str, list[str]] = {
    "billing_inquiry": [
        "bill", "billing", "invoice", "charge", "charged", "payment",
        "refund", "receipt", "overcharge", "price",
    ],
    "technical_support": [
        "error", "bug", "crash", "broken", "not working", "failed",
        "failure", "issue", "slow", "login", "log in", "password",
        "reset", "trouble",
    ],
    "sales_question": [
        "pricing", "plan", "quote", "demo", "trial", "buy", "purchase",
        "discount", "enterprise", "upgrade", "features",
    ],
    "account_update": [
        "update", "change", "email", "address", "phone", "profile",
        "account", "settings", "username",
    ],
    "order_status": [
        "order", "shipping", "delivery", "deliver", "track", "tracking",
        "package", "shipped", "arrive",
    ],
    "fallback": [],
}


@dataclass
class Classification:
    intent: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    needs_human: float = 0.0
    raw: str = ""


class IntentClassifier:
    def __init__(
        self,
        intents: list[Intent],
        *,
        backend: str = "mock",
        jev_api_key: str = "",
        jev_model: str = "jev-1.13.0",
        jev_api_url: str = "https://api.typesafe.ai/v1/systemone",
        jev_timeout_s: float = 10.0,
    ) -> None:
        if backend not in ("mock", "jev"):
            raise ValueError(f"unknown backend {backend!r}")
        self.intents = intents
        self.backend = backend
        self.jev: JevClient | None = None
        if backend == "jev":
            self.jev = JevClient(
                jev_api_key,
                model=jev_model,
                api_url=jev_api_url,
                timeout_s=jev_timeout_s,
            )

    # ------------------------------------------------------------------ API
    def classify(self, text: str) -> Classification:
        text = (text or "").strip()
        if not text:
            return Classification(
                intent="fallback", confidence=0.0, needs_human=0.0, raw="empty query"
            )
        if self.backend == "mock":
            return self._classify_mock(text)
        return self._classify_jev(text)

    # -------------------------------------------------------------- backends
    def _classify_mock(self, text: str) -> Classification:
        """Keyword scoring that mimics the shape of a Jev response.

        Confidence comes from the margin between the top two intents;
        a close race raises needs_human, like a Noul would.
        """
        lowered = text.lower()
        scores: dict[str, int] = {}
        for intent in self.intents:
            keywords = _MOCK_KEYWORDS.get(intent.name, [])
            scores[intent.name] = sum(1 for kw in keywords if kw in lowered)

        total = sum(scores.values())
        if total == 0:
            probs = {i.name: 1.0 / len(self.intents) for i in self.intents}
            return Classification(
                intent="fallback",
                confidence=0.25,
                probabilities=probs,
                needs_human=0.85,
                raw="mock: no keyword hits",
            )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_intent, top_hits = ranked[0]
        second_hits = ranked[1][1] if len(ranked) > 1 else 0
        probs = {name: hits / total for name, hits in scores.items()}

        margin = top_hits - second_hits
        confidence = min(0.95, 0.55 + 0.10 * margin + 0.03 * top_hits)
        # Close race with real signal on both sides -> a human should look.
        # A single keyword hit is weak but unambiguous: route it, don't escalate.
        needs_human = 0.75 if (margin < 2 and top_hits >= 2) else 0.15
        if top_intent == "fallback":
            needs_human = 0.85
        return Classification(
            intent=top_intent,
            confidence=round(confidence, 3),
            probabilities={k: round(v, 3) for k, v in probs.items()},
            needs_human=needs_human,
            raw=f"mock: {top_hits} keyword hits, margin {margin}",
        )

    def _classify_jev(self, text: str) -> Classification:
        assert self.jev is not None
        try:
            decision = self.jev.decide(text, self.intents)
        except JevError as exc:
            # Degrade to a low-confidence fallback classification instead of
            # failing the request: the router's confidence gate will send it
            # to the fallback worker.
            log.error("Jev classification failed: %s", exc)
            probs = {i.name: 1.0 / len(self.intents) for i in self.intents}
            return Classification(
                intent="fallback",
                confidence=0.2,
                probabilities=probs,
                needs_human=0.9,
                raw=f"jev error: {exc}",
            )
        log.info(
            "jev decision: intent=%s confidence=%.3f needs_human=%.3f latency=%.0fms model=%s",
            decision.intent,
            decision.confidence,
            decision.needs_human,
            decision.latency_ms,
            decision.model,
        )
        return Classification(
            intent=decision.intent,
            confidence=decision.confidence,
            probabilities=decision.probabilities,
            needs_human=decision.needs_human,
            raw=f"jev {decision.model} ({decision.latency_ms:.0f}ms)",
        )
