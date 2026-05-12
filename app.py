"""
Map LLM Recommendation API.

Endpoints:
  GET  /health          — smoke test, returns variant_id
  GET  /suggest-types   — Q2b: LLM maps intent → place type shortlist
  POST /recommend       — full pipeline: geocode → recall → filter → embed re-rank
  GET  /metrics         — rolling p50/p95/p99 latency over last 1000 requests
"""
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from config import cfg
from src.embedder import rank_by_similarity
from src.llm import init_vertex, suggest_place_types
from src.logger import StructuredLogger
from src.models import NearbySearchParams
from src.places import geocode, nearby_search

logger = StructuredLogger("api")

# Rolling window for p95/p99 — per instance; use Cloud Monitoring for fleet-wide aggregation
_latencies: deque = deque(maxlen=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_vertex()
    yield


app = FastAPI(title="Map LLM Recommendation API", lifespan=lifespan)


class LatencyMiddleware(BaseHTTPMiddleware):
    """
    Records end-to-end request latency for every route.
    p95/p99 — not average — is the right signal for LLM-backed endpoints
    where tail latency governs user experience (CLAUDE.md, Jeff Dean 2013).
    """
    async def dispatch(self, request: Request, call_next):
        t0 = time.time()
        response = await call_next(request)
        _latencies.append((time.time() - t0) * 1000)
        return response


app.add_middleware(LatencyMiddleware)


# ── Request / response schemas ────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str
    min_rating: float = Field(ge=1.0, le=5.0)
    max_rating: float = Field(ge=1.0, le=5.0)
    intent_text: str
    selected_types: list[str]
    city: str
    country: str
    description: str


class PlaceResult(BaseModel):
    name: str
    address: str
    rating: float | None
    user_rating_count: int | None
    price_level: str | None
    open_now: bool | None
    primary_type: str | None
    maps_url: str | None
    similarity_score: float
    review_snippets: int   # count only — full text is in the request log for audit


class RecommendResponse(BaseModel):
    request_id: str
    variant_id: str         # log this; without it metric changes can't be attributed
    query: str
    description: str
    results: list[PlaceResult]
    candidate_count: int    # places before embedding re-rank — monitors recall health
    total_latency_ms: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Smoke test. Cloud Run readiness probe hits this after every deploy."""
    return {"status": "ok", "variant_id": cfg.serving.variant_id}


@app.get("/suggest-types")
def suggest_types(intent: str):
    """
    Q2b: map a free-text intent to a shortlist of Google Places API types.
    Called by the client before submitting /recommend so the user can confirm types.
    """
    rid = uuid.uuid4().hex[:8]
    types = suggest_place_types(intent, rid)
    return {"request_id": rid, "intent": intent, "types": types}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    request_id = uuid.uuid4().hex[:8]

    api_key = os.getenv("PLACES_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="PLACES_API_KEY not configured on server.")

    if req.min_rating > req.max_rating:
        raise HTTPException(status_code=422, detail="min_rating must be ≤ max_rating.")

    # Log full inputs — HTTP 200 does not mean correct answers;
    # prediction logging is the only way to detect silent quality degradation (CLAUDE.md).
    logger.info(
        request_id, "request_start",
        variant_id=cfg.serving.variant_id,
        query=req.query,
        description=req.description,
        selected_types=req.selected_types,
        city=req.city,
        country=req.country,
        rating_range=[req.min_rating, req.max_rating],
    )
    t_start = time.time()

    # ── Geocode ───────────────────────────────────────────────────────────────
    location_name = f"{req.city}, {req.country}"
    try:
        lat, lng = geocode(location_name, api_key, request_id)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Geocoding failed: {exc}")

    # ── Recall: Nearby Search ─────────────────────────────────────────────────
    params = NearbySearchParams(
        location_name=location_name,
        latitude=lat,
        longitude=lng,
        radius_meters=cfg.recall.default_radius_meters,
        included_types=req.selected_types,
        keyword=None,
        max_results=cfg.places.max_candidates,
    )
    candidates = nearby_search(params, api_key, request_id)

    if not candidates:
        raise HTTPException(status_code=404, detail="No places found. Try different types or a wider radius.")

    # ── Rating filter ─────────────────────────────────────────────────────────
    in_range = [p for p in candidates if p.rating is not None and req.min_rating <= p.rating <= req.max_rating]
    unrated  = [p for p in candidates if p.rating is None]
    filtered = in_range + unrated

    if not filtered:
        raise HTTPException(status_code=404, detail=f"No places with rating {req.min_rating}–{req.max_rating}. Try widening the range.")

    # ── Stage 3: embedding re-rank ────────────────────────────────────────────
    ranked = rank_by_similarity(
        description=req.description,
        places=filtered,
        top_k=cfg.embedding.top_k,
        request_id=request_id,
    )

    total_ms = round((time.time() - t_start) * 1000, 1)

    results = [
        PlaceResult(
            name=p.name,
            address=p.address,
            rating=p.rating,
            user_rating_count=p.user_rating_count,
            price_level=p.price_level,
            open_now=p.open_now,
            primary_type=p.primary_type,
            maps_url=p.maps_url,
            similarity_score=round(score, 4),
            review_snippets=len(p.reviews),
        )
        for p, score in ranked
    ]

    logger.info(
        request_id, "request_complete",
        variant_id=cfg.serving.variant_id,
        candidate_count=len(candidates),
        filtered_count=len(filtered),
        result_names=[r.name for r in results],
        total_latency_ms=total_ms,
    )

    return RecommendResponse(
        request_id=request_id,
        variant_id=cfg.serving.variant_id,
        query=req.query,
        description=req.description,
        results=results,
        candidate_count=len(candidates),
        total_latency_ms=total_ms,
    )


@app.get("/metrics")
def metrics():
    """
    Rolling p50/p95/p99 latency over the last 1000 requests on this instance.
    For fleet-wide aggregation use Cloud Monitoring; this endpoint is for
    quick smoke-checks and post-deploy validation.
    """
    if not _latencies:
        return {"message": "no requests recorded yet"}
    data = sorted(_latencies)
    n = len(data)
    return {
        "variant_id": cfg.serving.variant_id,
        "request_count": n,
        "p50_ms": round(data[int(n * 0.50)], 1),
        "p95_ms": round(data[min(int(n * 0.95), n - 1)], 1),
        "p99_ms": round(data[min(int(n * 0.99), n - 1)], 1),
        "min_ms": round(data[0], 1),
        "max_ms": round(data[-1], 1),
    }
