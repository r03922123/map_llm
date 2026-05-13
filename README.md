# Map LLM

A place recommendation API powered by Google Places API and Vertex AI. Takes a natural-language description of what you're looking for, recalls nearby candidates, filters by rating, and re-ranks using embedding similarity between your description and place reviews.

## For New Team Members

This project is a production-grade implementation of the **recall → filter → rank** pattern used in every large-scale recommendation system at Google, Meta, LinkedIn, and Airbnb. If you have never built an ML-backed serving system before, read the docs in this order:

1. **This README** — system overview, API contract, deployment mechanics
2. **`docs/ml_methods.md`** — why each ML decision was made; starts with foundational concepts (embeddings, retrieval vs. generation) before assuming any ML background
3. **`docs/engineering.md`** — production engineering decisions: observability, retry strategy, config management, phased rollout

Each doc explains not just *what* was built but *why* — including what was deliberately left out and the trade-off that drove that choice. If you encounter a decision that seems arbitrary, look for the "Why" explanation nearby; if it is missing, that is a documentation gap worth flagging.

## Architecture

```
User query
    │
    ▼
Q1  Rating range (min / max)
Q2  Intent → LLM maps to Google Places types
Q3  City + country (geocoded deterministically)
Q4  Free-text ideal place description (embedded for ranking)
    │
    ▼
Geocode  ──────────────────────────────────  Google Maps Geocoding API
    │
    ▼
Recall   ──────────────────────────────────  Places Nearby Search (up to 20 candidates)
    │
    ▼
Filter   ──────────────────────────────────  Rating range (unrated places kept)
    │
    ▼
Re-rank  ──────────────────────────────────  text-embedding-004 cosine similarity
    │                                         (user description vs place name + reviews)
    ▼
Top-K results
```

**Models**
- LLM: `gemini-2.5-flash` via Vertex AI (type suggestion)
- Embeddings: `text-embedding-004` (768 dims) via Vertex AI

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Readiness probe — returns `variant_id` and Cloud Run `revision` |
| `GET` | `/suggest-types?intent=<text>` | Maps free-text intent to Google Places API types |
| `POST` | `/recommend` | Full pipeline: geocode → recall → filter → embed re-rank |
| `GET` | `/metrics` | Rolling p50/p95/p99 latency over last 1000 requests |

### POST /recommend

```json
{
  "query": "cozy ramen spot",
  "min_rating": 4.0,
  "max_rating": 5.0,
  "intent_text": "ramen restaurant",
  "selected_types": ["restaurant"],
  "city": "San Francisco",
  "country": "USA",
  "description": "quiet, warm lighting, rich tonkotsu broth, not too crowded"
}
```

## Setup

### Prerequisites

- Python 3.12+
- GCP project with these APIs enabled:
  - Vertex AI API
  - Places API (New)
  - Geocoding API
  - Cloud Run API
  - Secret Manager API
- `PLACES_API_KEY` stored in Secret Manager (see Deployment)

### Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and fill in credentials
cp .env.example .env   # set PLACES_API_KEY and GOOGLE_CLOUD_PROJECT

# Run CLI
python recommend.py

# Run API server
uvicorn app:app --reload --port 8080
```

### Environment variables

| Variable | Where | Description |
|----------|-------|-------------|
| `PLACES_API_KEY` | Secret Manager / `.env` | Google Maps API key |
| `GOOGLE_CLOUD_PROJECT` | `.env` / Cloud Run | GCP project ID |
| `K_REVISION` | Auto-set by Cloud Run | Revision name for snapshot versioning |

## Configuration

All tunables live in `config.yaml` — no code changes needed to tune:

```yaml
recall:
  default_radius_meters: 1000   # ~10 min walk; increase for suburban areas

embedding:
  top_k: 10                     # final results after re-rank

serving:
  variant_id: "v1"              # bump on every pipeline change for A/B attribution
```

## Deployment

### Deploy (phased rollout)

```bash
# Shadow → smoke test → 10% canary → manual gate
bash scripts/deploy.sh

# Skip manual gate, go straight to 100%
bash scripts/deploy.sh --full
```

The script:
1. Builds a git-SHA-tagged image and pushes to Artifact Registry
2. Deploys new revision at 0% traffic
3. Runs smoke test against the stable endpoint as a gate
4. Cuts 10% canary traffic
5. Prints promote / rollback commands

### Promote canary to full

```bash
gcloud run services update-traffic map-llm \
  --project=<PROJECT_ID> --region=us-central1 \
  --to-revisions=<REVISION>=100
```

### Smoke test

```bash
python scripts/smoke_test.py
# or against a specific URL:
python scripts/smoke_test.py https://your-service-url.run.app
```

### PLACES_API_KEY in Secret Manager

```bash
echo -n "YOUR_API_KEY" | gcloud secrets create PLACES_API_KEY --data-file=-
gcloud secrets add-iam-policy-binding PLACES_API_KEY \
  --member="serviceAccount:<SA>@<PROJECT>.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## Project structure

```
map_llm/
├── app.py                      # Entry point: re-exports app for uvicorn app:app
├── recommend.py                # Interactive CLI
├── config.yaml                 # All tunables (no code change needed to tune)
├── Dockerfile
├── requirements.txt
├── scripts/
│   ├── deploy.sh               # Phased rollout: shadow → canary → full
│   └── smoke_test.py           # Post-deploy gate
├── docs/
│   ├── engineering.md          # Engineering design decisions
│   └── ml_methods.md           # ML methods and production parallels
└── map_llm/                    # Main Python package
    ├── config.py               # Typed config dataclasses + cfg singleton
    ├── models.py               # Shared domain models (Place, UserIntent, etc.)
    ├── api/
    │   └── server.py           # FastAPI app: routes, schemas, middleware
    ├── llm/
    │   └── gemini.py           # Gemini: type suggestion + LLM re-ranker
    ├── observability/
    │   └── logger.py           # StructuredLogger — newline-delimited JSON
    └── pipeline/
        ├── recall.py           # Geocoding + Nearby Search (Places API)
        └── ranker.py           # Batch embed + cosine similarity ranking
```

## Production standards

- **Retry**: exponential backoff + jitter on all API calls (tenacity)
- **Latency**: p50/p95/p99 tracked per instance; `/metrics` endpoint for quick checks
- **Observability**: structured JSON logs with `request_id` (Dapper pattern) and `variant_id` on every request
- **Versioning**: `K_REVISION` in `/health` maps 1:1 to Docker image for rollback targeting
- **Deployment gate**: smoke test must pass before any revision receives traffic
