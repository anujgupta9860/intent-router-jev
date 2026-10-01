# Intent Router (Jev edition) — POC

An intent classifier that reads a user query, picks the owning intent, and
routes the query to the correct **worker agent over the A2A protocol** —
with the classification done by **Jev, TypeSafe AI's System One decision
model** instead of a generative LLM.

Same concept as the earlier Gemma-based `intent-router-poc`, but the
classifier is now a *typed decision* API: one request carries the query
(state) plus two questions — a `Choice` over intents and a `Noul`
("needs human review?") — and Jev returns probabilities + confidence in a
single parallel pass, in 70–500 ms, with no text generation to parse.

## Architecture

```
                        ┌──────────────────────────┐
                        │      User query          │
                        └────────────┬─────────────┘
                                     │ POST /route {"text": "..."}
                                     ▼
┌──────────────────────────────────────────────────────────────┐
│                 Intent Router (Jev) — FastAPI                  │
│                                                              │
│  ┌──────────────┐   ┌─────────────────────┐   ┌────────────┐ │
│  │ data/        │   │ IntentClassifier    │   │ RouterAgent│ │ │
│  │ intents.yaml │──▶│  backends:          │──▶│ confidence │ │ │
│  │ (taxonomy)   │   │  mock | jev         │   │ gate →     │ │ │
│  └──────────────┘   └─────────────────────┘   │ worker URL │ │
│                           │                   └─────┬──────┘ │
│                           ▼                         │ A2A    │
│                  ┌─────────────────┐                │ dispatch
│                  │ POST api.       │         ┌──────┴──────┐
│                  │ typesafe.ai/    │         ▼             ▼
│                  │ v1/systemone    │   worker-a        worker-b
└──────────────────┴────────────────┴─────────────────────────┘
   Choice(intent) + Noul(needs_human) → probabilities + confidence
```

Request flow inside the router (`src/`):

1. `app.py` — FastAPI service: `POST /classify` (intent + probabilities +
   confidence) and `POST /route` (full decision + A2A dispatch).
2. `jev_client.py` — `JevClient`: builds the two-question System One
   request from `data/intents.yaml` and POSTs it to
   `https://api.typesafe.ai/v1/systemone` (or a custom `TYPESAFE_API_URL`).
   Raw `httpx`, no SDK required.
3. `classifier.py` — `IntentClassifier.classify(text)` returns
   `Classification(intent, confidence, probabilities, needs_human)`.
   Backends: `mock` (deterministic keyword scoring shaped like a Jev
   response — zero deps, local dev / CI) and `jev` (live API).
4. `router.py` — `RouterAgent.handle_query(text)`: if confidence is below
   `CONFIDENCE_THRESHOLD` **or** the Noul probability exceeds
   `HUMAN_REVIEW_THRESHOLD`, the query goes to the `fallback` worker;
   otherwise the worker URL is resolved from `WORKER_AGENTS_JSON` and the
   query is dispatched.
5. `a2a_client.py` — `A2AClient.dispatch(agent_url, text, intent)`: uses the
   official `a2a-sdk` when installed, otherwise a documented stub that logs
   the dispatch and returns a synthetic task id.

## Why Jev instead of an LLM classifier?

| | LLM classifier (Gemma prompt) | Jev classifier |
|---|---|---|
| Output | free text → parse JSON, validate enums, repair drift | typed `choice` + `noul` — closed output space, no parsing |
| Confidence | self-reported, uncalibrated | calibrated (≥0.9 → right 94–96% in independent tests) |
| Latency | seconds | 70–500 ms end-to-end |
| Cost | per-token generation | $0.042 / M input tokens, output free |
| Impossible | model invents an intent not in your list | can't happen — options are yours |

See `../your_files/jev-agent-guide.md` for the full Jev deep-dive.

## Local quickstart

Prerequisites: Python 3.11+, Docker (for the compose demo).

```bash
cd intent-router-jev
pip install -r requirements.txt

# 1) Mock mode — no API key, deterministic demo
CLASSIFIER_BACKEND=mock uvicorn src.app:app --port 8080

# 2) Live Jev mode (early access — get a key at console.typesafe.ai)
export TYPESAFE_API_KEY="ts_..."
CLASSIFIER_BACKEND=jev uvicorn src.app:app --port 8080
```

Try it:

```bash
curl -s localhost:8080/health
curl -s -X POST localhost:8080/classify \
  -H 'content-type: application/json' \
  -d '{"text": "I was charged twice on my last invoice"}'
# {"intent":"billing_inquiry","confidence":0.94,
#  "probabilities":{"billing_inquiry":0.94,...},"needs_human":0.12}

curl -s -X POST localhost:8080/route \
  -H 'content-type: application/json' \
  -H 'content-type: application/json' \
  -d '{"text": "Where is my order?"}'
```

### Full local demo (router + two stub workers)

```bash
docker compose up --build
# router: http://localhost:8080, worker-a: :8001, worker-b: :8002
```

With `CLASSIFIER_BACKEND=jev` and `TYPESAFE_API_KEY` set on the router
service, the compose demo exercises the live API end to end.

### Tests

```bash
pytest eval/ -q   # no network, no optional deps
```

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `CLASSIFIER_BACKEND` | `mock` | `mock` (keyword demo) or `jev` (live API) |
| `TYPESAFE_API_KEY` | — | API key for the `jev` backend (Secret Manager / K8s Secret in prod) |
| `TYPESAFE_MODEL` | `jev-1.13.0` | Model id; pin a version, `jev-latest` moves |
| `TYPESAFE_API_URL` | `https://api.typesafe.ai/v1/systemone` | Override for proxies / future regions |
| `TYPESAFE_TIMEOUT_S` | `10` | Per-request timeout |
| `CONFIDENCE_THRESHOLD` | `0.6` | Below this → route to `fallback` worker |
| `HUMAN_REVIEW_THRESHOLD` | `0.5` | Noul "needs human" probability above this → `fallback` worker |
| `WORKER_AGENTS_JSON` | built-in map | intent → A2A worker base URL |
| `LOG_LEVEL` | `INFO` | — |

Add an intent without touching code: add a block to `data/intents.yaml`
(name, description, worker, examples) — the description becomes the
`Choice` criterion text Jev sees, so write it like you'd explain the
category to a smart colleague. Keep each description one-dimensional;
Jev adapts to new categories with no fine-tuning.

## Deploying to GCP

Nothing model-side to deploy — Jev is a hosted API. You deploy *this
service* and give it the key.

**Cloud Run (recommended for the router):**

```bash
# store the key once
echo -n "$TYPESAFE_API_KEY" | gcloud secrets create typesafe-api-key --data-file=-

gcloud run deploy intent-router-jev \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars CLASSIFIER_BACKEND=jev,TYPESAFE_MODEL=jev-1.13.0,CONFIDENCE_THRESHOLD=0.6 \
  --set-secrets TYPESAFE_API_KEY=typesafe-api-key:latest
```

The call is outbound HTTPS (70–500 ms); min-instances 0 is fine. The
provider's limits (250k tokens/sec, 1,200 req/min) bind before GCP's do.

**GKE:** `helm/intent-router-jev/` mirrors the earlier chart — ConfigMap
for config, Secret for `TYPESAFE_API_KEY` (prefer a kubectl-created
secret or External Secrets in production; see `templates/NOTES.txt`).

```bash
helm upgrade --install intent-router-jev ./helm/intent-router-jev \
  --set secrets.typesafeApiKey="$TYPESAFE_API_KEY"
```

## Repo layout

```
├── data/intents.yaml        # intent taxonomy (also the Choice criteria)
├── src/
│   ├── app.py               # FastAPI: /health /classify /route
│   ├── classifier.py        # IntentClassifier: mock | jev backends
│   ├── jev_client.py        # System One API client (httpx)
│   ├── router.py            # confidence gate + Noul gate + A2A dispatch
│   ├── a2a_client.py        # A2A dispatch (a2a-sdk or stub)
│   ├── config.py            # env-driven Settings
│   ├── intents.py           # taxonomy loader
│   └── workers/stub_worker.py
├── eval/test_router.py      # unit tests (no network)
├── Dockerfile  docker-compose.yml
├── helm/intent-router-jev/  # GKE chart
└── scripts/                 # GCP helpers
```

## Production notes

- **Pin the model version** (`TYPESAFE_MODEL=jev-1.13.0`); log the resolved
  `model` from each response in your telemetry.
- **429/529** → exponential backoff (already in `jev_client.py`); the
  classifier degrades to a low-confidence `fallback` classification rather
  than failing the request.
- **Thresholds are yours to tune**: run `eval/` examples through both
  backends, look at the confidence distribution, then set
  `CONFIDENCE_THRESHOLD` / `HUMAN_REVIEW_THRESHOLD` from data.
- TypeSafe publishes a "Jev 1.13 jaggedness" page of known rough edges —
  read it before production. Text input only; English is strongest.
