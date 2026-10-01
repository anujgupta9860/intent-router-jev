"""Unit tests for the Jev intent router (no network, no optional deps)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.classifier import IntentClassifier
from src.config import Settings
from src.intents import load_intents
from src.jev_client import JevClient, JevError, build_questions, parse_decision
from src.router import RouterAgent


@pytest.fixture()
def intents():
    return load_intents()


@pytest.fixture()
def router(intents):
    clf = IntentClassifier(intents, backend="mock")
    workers = {
        "billing_inquiry": "http://billing:9999",
        "fallback": "http://fallback:9999",
    }
    return RouterAgent(clf, workers, confidence_threshold=0.6,
                       human_review_threshold=0.5)


# ------------------------------------------------------------------ config
def test_config_defaults(monkeypatch):
    for var in (
        "CLASSIFIER_BACKEND", "TYPESAFE_API_KEY", "TYPESAFE_MODEL",
        "TYPESAFE_API_URL", "TYPESAFE_TIMEOUT_S", "CONFIDENCE_THRESHOLD",
        "HUMAN_REVIEW_THRESHOLD", "WORKER_AGENTS_JSON", "LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.classifier_backend == "mock"
    assert s.typesafe_model == "jev-1.13.0"
    assert s.typesafe_api_url == "https://api.typesafe.ai/v1/systemone"
    assert s.confidence_threshold == 0.6
    assert s.human_review_threshold == 0.5
    assert s.worker_agents["fallback"].startswith("http")


def test_config_rejects_bad_backend(monkeypatch):
    monkeypatch.setenv("CLASSIFIER_BACKEND", "gpt")
    with pytest.raises(ValueError):
        Settings()


def test_config_requires_key_for_jev(monkeypatch):
    monkeypatch.setenv("CLASSIFIER_BACKEND", "jev")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        Settings()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_test")
    assert Settings().classifier_backend == "jev"


def test_config_worker_agents_json(monkeypatch):
    monkeypatch.setenv(
        "WORKER_AGENTS_JSON", '{"a": "http://x:1", "fallback": "http://y:2"}'
    )
    s = Settings()
    assert s.worker_url("a") == "http://x:1"
    assert s.worker_url("missing") == "http://y:2"


# ------------------------------------------------------------ jev client
def test_build_questions_uses_taxonomy(intents):
    q = build_questions(intents)
    assert set(q) == {"intent", "human_review"}
    assert q["intent"]["type"] == "choice"
    assert q["human_review"]["type"] == "noul"
    criteria = q["intent"]["criteria"]
    assert criteria["billing_inquiry"].startswith("Questions about bills")
    # every taxonomy intent is offered; the model can never invent one
    assert set(criteria) == {i.name for i in intents}


def test_parse_decision(intents):
    payload = {
        "model": "jev-1.13.0",
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "billing_inquiry",
                "probabilities": {
                    "billing_inquiry": 0.94,
                    "technical_support": 0.04,
                    "fallback": 0.02,
                },
                "confidence": 0.91,
            },
            "human_review": {"type": "noul", "noul": 0.12},
        },
        "usage": {"input_tokens": 180, "output_tokens": 0},
    }
    d = parse_decision(payload, intents, latency_ms=120.0)
    assert d.intent == "billing_inquiry"
    assert d.confidence == pytest.approx(0.91)
    assert d.needs_human == pytest.approx(0.12)
    assert d.latency_ms == pytest.approx(120.0)
    assert d.usage["output_tokens"] == 0


def test_parse_decision_unknown_intent_falls_back(intents):
    payload = {
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "made_up_intent",
                "probabilities": {},
                "confidence": 0.99,
            },
            "human_review": {"type": "noul", "noul": 0.0},
        }
    }
    d = parse_decision(payload, intents)
    assert d.intent == "fallback"


def test_jev_client_needs_key():
    with pytest.raises(ValueError):
        JevClient("")


def test_jev_client_401_raises(monkeypatch, intents):
    import httpx

    class FakeResp:
        status_code = 401
        text = "unauthorized"

        def json(self):
            return {}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
    client = JevClient("bad-key")
    with pytest.raises(JevError, match="401"):
        client.decide("hello", intents)


# -------------------------------------------------------------- classifier
def test_mock_classify_billing(intents):
    clf = IntentClassifier(intents, backend="mock")
    c = clf.classify("I was charged twice on my last invoice")
    assert c.intent == "billing_inquiry"
    assert c.confidence > 0.6
    assert abs(sum(c.probabilities.values()) - 1.0) < 1e-6


def test_mock_classify_empty(intents):
    clf = IntentClassifier(intents, backend="mock")
    c = clf.classify("   ")
    assert c.intent == "fallback"
    assert c.confidence == 0.0


def test_mock_ambiguous_raises_needs_human(intents):
    clf = IntentClassifier(intents, backend="mock")
    # hits both billing and order keywords -> close race
    c = clf.classify("my payment for the order is late, track my refund")
    assert c.needs_human > 0.5


def test_classifier_rejects_unknown_backend(intents):
    with pytest.raises(ValueError):
        IntentClassifier(intents, backend="llama")


# ------------------------------------------------------------------ router
def test_high_confidence_routes_to_worker(router):
    out = router.handle_query("Why was I charged twice on my invoice?")
    assert out["intent"] == "billing_inquiry"
    assert out["worker"] == "http://billing:9999"
    assert out["routed_to_fallback"] is False
    assert out["task_id"].startswith("task-")


def test_low_confidence_gates_to_fallback(router):
    out = router.handle_query("Tell me a joke")
    assert out["intent"] == "fallback"
    assert out["worker"] == "http://fallback:9999"
    assert out["gates"]["low_confidence"] is True


def test_human_review_gate(router):
    out = router.handle_query("my payment for the order is late, track my refund")
    # ambiguous query trips the human gate
    assert out["gates"]["human_review"] is True
    assert out["intent"] == "fallback"
    assert out["worker"] == "http://fallback:9999"


def test_resolve_worker_falls_back(router):
    assert router.resolve_worker("nope") == "http://fallback:9999"
