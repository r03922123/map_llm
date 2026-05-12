# Engineering Design

## Overview

Map LLM is a place recommendation service. It exposes a REST API (FastAPI on Cloud Run) and an interactive CLI. A request flows through three sequential stages — recall, filter, rank — each calling a separate external service. All stages are deterministic except the LLM type-suggestion call in the question phase.

```
Client
  │
  ▼
POST /recommend
  │
  ├─ Geocode           Google Maps Geocoding API    → lat/lng
  ├─ Recall            Places Nearby Search API     → up to 20 candidates
  ├─ Rating filter     in-process                   → narrows candidate set
  └─ Embedding re-rank Vertex AI text-embedding-004 → top-K results
```

---

## API Design

**FastAPI** is used for the HTTP layer. Four endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Readiness probe — returns `status`, `variant_id`, `revision` |
| `GET` | `/suggest-types?intent=<text>` | LLM call: maps free-text intent to Places API types |
| `POST` | `/recommend` | Full pipeline |
| `GET` | `/metrics` | Rolling p50/p95/p99 latency |

### Why separate `/suggest-types`

The type-suggestion LLM call is extracted as its own endpoint so the client can show the user a confirmation step ("Did you mean: restaurant, ramen_restaurant?") before committing to the full pipeline. This matters because the Places API charges per request and incorrect types silently return zero results rather than an error.

### Request / response contracts

All inputs are validated by Pydantic at the boundary. `min_rating` and `max_rating` use `Field(ge=1.0, le=5.0)` — Pydantic raises 422 before the handler runs, so no defensive checks are needed inside the business logic. One manual check is kept: `min_rating > max_rating` is a valid-value cross-field constraint that Pydantic field validators don't cover without a `model_validator`.

`PlaceResult.review_snippets` is an integer count, not the review text itself. Full review text is in the request log for audit. This avoids surfacing PII-adjacent content in the API response while keeping it available for quality review.

---

## Request Lifecycle

```
app.py: recommend()
  │
  ├── logger.info(request_id, "request_start", ...)     ← all inputs logged before anything runs
  │
  ├── geocode(location_name, api_key, request_id)
  │     └── GET maps.googleapis.com/maps/api/geocode/json
  │
  ├── nearby_search(params, api_key, request_id)
  │     └── POST places.googleapis.com/v1/places:searchNearby
  │
  ├── rating filter   [in-process, no I/O]
  │
  ├── rank_by_similarity(description, places, top_k, request_id)
  │     └── POST Vertex AI embed_content   (single batch call)
  │
  └── logger.info(request_id, "request_complete", ...)  ← result names + latency logged
```

`request_id` is a `uuid4().hex[:8]` generated at the top of the handler and passed to every function call. Every log line emitted by any module carries it. This is the Dapper pattern: a single grep on `request_id` reconstructs the full trace for any request without a distributed tracing system.

---

## Observability

### Structured logging

`StructuredLogger` emits newline-delimited JSON. Each record has a fixed schema:

```json
{"ts": 1715000000.123, "level": "INFO", "request_id": "a1b2c3d4", "event": "nearby_search_done", "candidate_count": 18, "with_reviews": 14, "latency_ms": 312.4}
```

Fixed keys (`ts`, `level`, `request_id`, `event`) make logs queryable in Cloud Logging with a simple filter:
```
jsonPayload.request_id="a1b2c3d4"
```

Variable fields are kwargs passed by the caller, keeping the logger itself generic.

### What is logged and why

| Event | Module | Why |
|-------|--------|-----|
| `request_start` | app.py | Full inputs — HTTP 200 does not mean correct results; prediction logging is the only way to detect quality drift |
| `geocode_done` | places.py | lat/lng + latency — confirms the location resolved correctly |
| `nearby_search_done` | places.py | `candidate_count` + `with_reviews` — monitors recall health; a drop in `with_reviews` means the Preferred SKU field may have stopped returning data |
| `embed_done` | embedder.py | vector count + dims — confirms batch size and model dimensionality match |
| `similarity_ranked` | embedder.py | Top scores — detects score collapse (all scores near 0 means embedding quality has degraded) |
| `request_complete` | app.py | `result_names`, `total_latency_ms`, `variant_id` — end-to-end audit trail |

### Latency tracking

`LatencyMiddleware` wraps every request in `time.time()` and appends elapsed milliseconds to a `deque(maxlen=1000)`. `/metrics` reports p50/p95/p99 from that window.

**Why p95/p99, not mean:** LLM inference latency has high variance. A 200ms median with a 4000ms p99 means 1 in 100 users waits 20× longer. The mean would be ~220ms and would show nothing wrong. Tail latency governs user experience on LLM-backed endpoints (Jeff Dean, "The Tail at Scale," CACM 2013).

The deque is per-instance. For fleet-wide aggregates, use Cloud Monitoring; this endpoint is for quick post-deploy checks.

---

## Reliability

### Retry strategy

All three external calls (geocode, nearby_search, embed_texts) are decorated with:

```python
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10), reraise=True)
```

`wait_random_exponential` adds full jitter to the backoff. **Why jitter:** without it, retries from concurrent requests after a service hiccup are synchronised — all clients retry at the same intervals, creating a thundering herd that can take down a struggling upstream. Jitter de-correlates retries across clients (AWS Architecture Blog, "Exponential Backoff And Jitter," Marc Brooker, 2015).

`reraise=True` means the original exception propagates after 3 failures rather than wrapping it, preserving the original stack trace for debugging.

### Error handling

- Geocode failure → HTTP 422 (client error: the location string is the input, not a server fault)
- No candidates → HTTP 404 with actionable message ("try different types or wider radius")
- No candidates after rating filter → HTTP 404 with the specific constraint that failed
- `PLACES_API_KEY` missing → HTTP 500 (server misconfiguration, not client fault)

---

## Configuration

`config.yaml` holds all tunables. `config.py` loads it into typed Python dataclasses at import time (`cfg` singleton). No magic strings or `os.getenv` calls appear in business logic — all values come from `cfg`.

**Why typed dataclasses over a plain dict:** type errors on config access are caught at development time (e.g. `cfg.embedding.top_k` vs `cfg["embedding"]["top_k"]`). A typo in a dict key raises `KeyError` at runtime; a typo in an attribute raises `AttributeError` immediately on the first request.

**Why YAML over environment variables for tunables:** tunables like `top_k`, `radius_meters`, `temperature` change together as a unit during experiments. Grouping them in one file lets you version them as a snapshot. Environment variables are appropriate for secrets and deployment-time values (`PLACES_API_KEY`, `K_REVISION`) — not for ML hyperparameters.

---

## Secrets Management

`PLACES_API_KEY` is stored in GCP Secret Manager and mounted as an environment variable into the Cloud Run service. It is **not** in source code, `config.yaml`, or the Docker image.

```bash
# How it was set up:
gcloud secrets create PLACES_API_KEY --data-file=-   # pipe key via stdin
gcloud run services update map-llm \
  --set-secrets="PLACES_API_KEY=PLACES_API_KEY:latest"
```

The service account running Cloud Run must have `roles/secretmanager.secretAccessor`. If the secret is missing, the service starts but `/recommend` returns HTTP 500 immediately — it does not silently return empty results.

---

## Deployment

### Versioned images

Every deploy tags the Docker image with the git SHA:

```
us-central1-docker.pkg.dev/<PROJECT>/cloud-run-source-deploy/map-llm:<GIT_SHA>
```

This means every Cloud Run revision maps 1:1 to a specific commit. If quality regresses, the rollback command is:
```bash
gcloud run services update-traffic map-llm --to-revisions=<PREVIOUS_REVISION>=100
```
No rebuild required — the old image is still in Artifact Registry.

Cloud Run also sets `K_REVISION` automatically on each revision (e.g. `map-llm-00004-67r`). This is exposed in `/health` and `/metrics` so you can confirm which revision is actually serving without looking at the console.

### Phased rollout (`scripts/deploy.sh`)

```
[1] Build & push git-SHA-tagged image
[2] Deploy new revision at 0% traffic
[3] Smoke test against the stable endpoint  ← gate: exit 1 blocks traffic cut
[4] Route 10% traffic to new revision (canary)
[5] Pause for manual inspection → promote or rollback
```

**Why 0% first, then canary:** deploying at 0% traffic lets the new revision start up and pass the smoke test before any real user sees it. This catches startup failures (bad secret binding, import errors, missing config) that unit tests don't cover.

**Why 10% canary before 100%:** the smoke test uses a synthetic payload. Real traffic has query distributions the smoke test cannot replicate. A 10% canary exposes the new revision to real diversity at limited blast radius, letting you measure error rates and latency before full promotion.

### Smoke test (`scripts/smoke_test.py`)

Runs against the live endpoint after every deploy. Checks:
- `/health`: HTTP 200, `status=ok`, `variant_id` present, `revision` present
- `/suggest-types`: HTTP 200, non-empty types array
- `/recommend`: HTTP 200, `request_id` / `variant_id` / `results` / `candidate_count` present

Exits 0 on pass, 1 on any failure. Designed to be a CI gate — a non-zero exit blocks the pipeline.

---

## Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

`requirements.txt` is copied before `COPY . .` to exploit Docker layer caching — dependency installation (the slow step) is only re-run when `requirements.txt` changes, not on every code change.

`.dockerignore` excludes `.venv/`, `.env`, `.git/`, `__pycache__` — the venv is the largest exclusion and prevents accidentally shipping a platform-specific local venv into a Linux container.

---

## Project Structure

```
map_llm/
├── app.py              FastAPI service — routes, middleware, schemas
├── recommend.py        Interactive CLI — question phase + pipeline
├── config.py           Typed config dataclasses + cfg singleton
├── config.yaml         All tunables (no code change needed to tune)
├── Dockerfile
├── requirements.txt
├── scripts/
│   ├── deploy.sh       Phased rollout: shadow → canary → full
│   └── smoke_test.py   Post-deploy gate
└── src/
    ├── embedder.py     Batch embed + cosine similarity ranking
    ├── llm.py          Gemini: type suggestion + LLM re-ranker (unused in main pipeline)
    ├── logger.py       StructuredLogger — newline-delimited JSON
    ├── models.py       Pydantic models: Place, UserIntent, NearbySearchParams, Recommendation
    └── places.py       Geocoding + Nearby Search (Places API)
```
