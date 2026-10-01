# Software Design Review (SDR): Intent Router — Jev Edition

| | |
|---|---|
| **Author** | GoGo (for Anuj Gupta) |
| **Date** | 2026-10-01 |
| **Status** | Draft — POC built, awaiting review |
| **Repo** | `intent-router-jev` (local; GitHub push pending) |
| **Predecessor** | `intent-router-poc` (Gemma/LLM-based classifier, same routing concept) |

## 1. Problem statement

Every user query entering our support/agent surface must be sent to the
worker agent that owns it (billing, support, sales, account, orders) — or
to a human when it is risky or ambiguous. The previous POC did this with a
generative LLM (Gemma) prompted to return JSON. That works, but in
production it has three structural weaknesses:

1. **Brittle output.** The model generates free text; we parse JSON back
   out and validate enums. Any format drift breaks routing.
2. **Uncalibrated confidence.** The "confidence" number is self-reported
   prose — it cannot be trusted to gate actions.
3. **Cost/latency.** Every query pays for a full generation cycle even
   though the job only needs a label.

## 2. Goals

- G1. Route a user query to the correct worker agent over **A2A** with a
  typed, machine-consumable classification — no text parsing.
- G2. Make **two** decisions per query, not one: *which intent* (Choice)
  and *does a human need to look* (Noul).
- G3. Gate automatic routing on **calibrated** confidence: below threshold
  → fallback worker (human queue), with the reason recorded.
- G4. Answer in **< 500 ms** at roughly **$0.042 / M input tokens** (Jev
  pricing), output tokens free.
- G5. Add a new intent by editing `data/intents.yaml` — no prompt surgery,
  no fine-tuning, no redeploy of model weights (there are none).
- G6. Deploy the router on GCP (Cloud Run or GKE) calling Jev's hosted API;
  secrets via Secret Manager / K8s Secret.

## 3. Non-goals

- NG1. Jev does not replace the worker agents or the answering LLM — it is
  the *decision primitive inside* the loop, not the reasoner.
- NG2. No self-hosting of Jev (closed weights, hosted API only).
- NG3. No image/audio input (Jev is text-only); no multilingual quality
  claims without workload testing.
- NG4. This SDR does not cover the workers' internal logic — they are stub
  agents with A2A cards in the POC.

## 4. Design

### 4.1 Architecture

```
User query ──▶ POST /route ──▶ IntentClassifier (mock | jev)
                                     │
                     ┌───────────────┴───────────────┐
                     │ Jev System One request        │
                     │ state={query}                 │
                     │ Q1 Choice(intent): criteria = │
                     │    intents.yaml descriptions  │
                     │ Q2 Noul(human_review)         │
                     └───────────────┬───────────────┘
                                     │ 70–500 ms
                    ┌────────────────┴────────────────┐
                    │ RouterAgent dual gate           │
                    │ conf < 0.6  → fallback         │
                    │ human_p > 0.5 → fallback        │
                    └────────────────┬────────────────┘
                                     │ A2A dispatch
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
              worker-a :8001                    worker-b :8002
```

### 4.2 Why these two questions

- **Choice over intents.** The option set is closed and caller-defined, so
  Jev *cannot* invent an intent name. Per-option probabilities let us
  detect close races in code later (margin analysis) if we want a third
  signal.
- **Noul for human review.** Confidence answers "should I trust the
  label?"; the Noul answers "is this risky regardless of the label?"
  (e.g. a $480k payment instruction classified confidently as
  `billing_inquiry` still deserves a human). Two independent axes, both
  thresholded in code.

### 4.3 Request / response contract

Request (`POST https://api.typesafe.ai/v1/systemone`):

```json
{
  "model": "jev-1.13.0",
  "state": {"query": "<user text>"},
  "questions": {
    "intent": {"type": "choice", "instructions": "...", "criteria": {"<intent>": "<description>", "...": "..."}},
    "human_review": {"type": "noul", "instructions": "..."}
  }
}
```

Response is parsed defensively (`jev_client.parse_decision`): unknown
intent names → `fallback`; probabilities filtered to the taxonomy;
confidence and Noul clamped to [0, 1]. Question IDs are ours and are
never sent to the model.

### 4.4 Routing decision table

| Choice confidence | Noul human_p | Result |
|---|---|---|
| ≥ 0.6 | ≤ 0.5 | route to intent's worker |
| < 0.6 | ≤ 0.5 | fallback worker (`low_confidence` gate) |
| ≥ 0.6 | > 0.5 | fallback worker (`human_review` gate) |
| < 0.6 | > 0.5 | fallback worker (both gates) |

The `/route` response includes `gates: {low_confidence, human_review}` so
threshold tuning is data-driven (see §7).

### 4.5 Failure handling

- **401** → fail fast with a clear message (bad `TYPESAFE_API_KEY`).
- **422** → fail fast (malformed question — a code bug, not retryable).
- **408 / 429 / 529** → exponential backoff (0.5s, 1s, 2s), then degrade.
- **Any failure after retries** → low-confidence `fallback`
  classification (`needs_human=0.9`), which the gates route to the
  fallback worker. A Jev outage never drops a query; it escalates it.

### 4.6 Taxonomy as configuration

`data/intents.yaml` is the single source of truth: each intent's
`description` becomes the Choice criterion text Jev sees. Writing guidance:
one-dimensional, concrete, contrasting with siblings
("Questions about bills, invoices, charges, payments, refunds, or
receipts."). New intent = new YAML block → works immediately, zero model
work.

## 5. Alternatives considered

| Option | Verdict |
|---|---|
| Gemma LLM classifier (previous POC) | Works; kept as reference. Rejected for prod: brittle JSON parsing, uncalibrated confidence, seconds of latency. |
| Jev (this design) | **Chosen.** Typed output, calibrated confidence, 70–500 ms, $0.042/M in / free out. Trade-off: hosted-only, early-access, no SLA yet. |
| OpenAI Decisions API (previewed Sep 30, 2026) | Same job, different architecture. Worth a bake-off before committing — but Jev has live production references (Vercel, Bryo AI) today. |
| Cheap LLM (e.g. Gemini Flash) as classifier | Tied Jev on accuracy in independent tests. Viable fallback; Jev wins on calibration + latency, not raw accuracy. |

## 6. Security

- `TYPESAFE_API_KEY` never in the image: Secret Manager (Cloud Run) or
  K8s Secret / External Secrets (GKE). The Helm chart renders the Secret
  from `--set secrets.typesafeApiKey=...` and refuses to recreate it on
  upgrade (lookup guard).
- Queries sent to a US-hosted third-party API: PII policy applies — the
  `state` should carry the query text only, not account payloads, until
  data-processing terms are signed. TypeSafe states requests/responses
  are not used for training without consent; that is narrower than zero
  retention — confirm with legal for regulated data.

## 7. Testing & evaluation

- **Unit** (`eval/test_router.py`, 17 tests): config validation, question
  building from taxonomy, response parsing (incl. unknown-intent
  fallback), mock classification, both gates, worker resolution. No
  network, no optional deps.
- **Smoke** (`docker compose up`): router + two stub workers; verified
  routing, confidence gating, and human-review gating on sample queries.
- **Recommended before prod**: run the labeled examples in
  `data/intents.yaml` through the live `jev` backend; plot the confidence
  and Noul distributions; set `CONFIDENCE_THRESHOLD` /
  `HUMAN_REVIEW_THRESHOLD` from the curves, not from defaults. Re-run on
  every `TYPESAFE_MODEL` bump (`jev-latest` moves — pin versions).

## 8. Deployment (GCP)

- **Cloud Run** (recommended): container from `Dockerfile`,
  `CLASSIFIER_BACKEND=jev`, `TYPESAFE_API_KEY` from Secret Manager,
  `TYPESAFE_MODEL=jev-1.13.0` pinned. Outbound HTTPS only; min-instances
  0 is fine. Provider limits (250k tok/s, 1,200 req/min) bind first.
- **GKE**: `helm/intent-router-jev/` — ConfigMap for config, Secret for
  the key, `/health` probes, non-root + read-only filesystem.
- **Cost envelope** (illustrative): 1M queries/day × ~200 input tokens =
  200M tokens/day → **~$8.40/day** in Jev input cost. Output free.

## 9. Rollout plan

1. Ship behind the `mock` backend to validate routing/wiring (done in POC).
2. Enable `jev` backend on a shadow percentage; log decisions without
   acting; compare against current classifier.
3. Tune thresholds from shadow data (§7).
4. Cut over per-intent, keeping the fallback worker as the safety net.
5. Monitor: Jev latency p50/p99, gate trip rates, 429/529 rate, resolved
   model version per response.

## 10. Open questions

- OQ1. Early-access terms: SLA and rate-limit increases for production
  volume — confirm with TypeSafe.
- OQ2. Data residency: US-hosted API only today; acceptable for our query
  data?
- OQ3. Bake-off vs OpenAI Decisions API before the contract decision.
- OQ4. Should low-confidence queries go to a stronger LLM for a second
  opinion instead of straight to human? (Cost/quality trade-off to measure
  in shadow mode.)
- OQ5. GitHub PAT for the repo push is pending — repo currently local only.

## 11. Risks

- R1. **Vendor dependence**: single hosted API, no self-host/VPC option.
  Mitigation: `mock`-shaped fallback path; a cheap-LLM backend can be
  added behind the same `IntentClassifier` interface.
- R2. **Calibration drift** across model versions. Mitigation: pin
  `TYPESAFE_MODEL`, log resolved version, re-run eval on bump.
- R3. **Jev 1.13 known rough edges** (vendor-published "jaggedness" page).
  Mitigation: read before prod; shadow mode first.
