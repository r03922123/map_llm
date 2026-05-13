# Inherits: ABC ML System Design
# Location: /Users/linjiehong/Desktop/muthur/CLAUDE.md
@/Users/linjiehong/Desktop/muthur/CODE_STYLE.md

---

## Abstract Properties

```
serving_mode: real-time
sla_p99_ms: 2000
primary_eval_metric: ndcg@5
```

`sla_p99_ms = 2000`: three sequential external calls (geocode ~100ms, Nearby Search ~300ms, Vertex AI embed ~300ms) plus overhead. The 2000ms budget is the observed p99 ceiling at 1x load; breach triggers rollback.

`primary_eval_metric = ndcg@5`: measures whether known-good places appear near the top of the embedding re-rank output. NDCG is the correct metric here because position matters — a known-good place ranked #1 is better than the same place ranked #5. Accuracy (did any expected place appear?) ignores rank and is too weak a signal.

---

## Abstract Methods

### eval_strategy

**Golden set:** `data/golden/golden_set.json` — a frozen list of `(description, city, selected_types, expected_names)` tuples. Collected by running the live pipeline on diverse queries and manually verifying the top results are correct. Never used for hyperparameter tuning. Refresh when `build_place_text()` changes, the embedding model changes, or the set falls below 20 examples.

**Primary metric:** NDCG@5. A place is relevant (score 1) if its name appears in `expected_names`; irrelevant (score 0) otherwise. Ideal ranking has all expected places in positions 1–5.

**Promotion threshold:** NDCG@5 must not drop more than 5% from the current baseline (`data/golden/baseline.json`). If it does, the eval runner exits 1 and `deploy.sh` blocks the canary step.

**Eval runner:** `map_llm/evaluation/eval.py` — loads the golden set, runs the recall → filter → rank pipeline for each entry, computes per-query NDCG@5, reports mean, compares against baseline.

---

### feature_pipeline

**Data source:** Google Places API (Nearby Search, Preferred SKU) — fetched live at inference time, not a static dataset.

**Feature computation:** `map_llm/pipeline/ranker.py:build_place_text()` is the single source of truth for document representation. It concatenates `primary_type + name + address + reviews` (up to 5 snippets). Both the API path (`map_llm/api/server.py`) and the CLI (`recommend.py`) call this function — no duplication.

**Skew risk:** No trained model, so there is no training-serving skew in the traditional sense. The analogous risk is embedding-model drift: if `text-embedding-004` version at serve time differs from the version that produced any cached embeddings, cosine similarity becomes meaningless. Mitigation: embeddings are never cached — all are computed fresh per request. Version is pinned in `config.yaml`.

**Schema:** The document fields are `primary_type`, `name`, `address`, `reviews`. Any change to this set is a breaking change — the golden set must be re-run and a new baseline recorded before the change is promoted.

---

### rollback_trigger

| Signal | Threshold | Window | Action |
|---|---|---|---|
| Smoke test | any failure | post-deploy | stay at 0% traffic; do not cut canary |
| p99 latency | > 2000ms | 5-minute rolling window | roll back to previous revision |
| NDCG@5 | > 5% drop from baseline | eval run in `deploy.sh` | block canary step; do not build or deploy |

Latency alert: Cloud Monitoring on the Cloud Run service, alarm on `request_latencies` p99.
NDCG@5 gate: enforced by `scripts/deploy.sh` step [0] before image build.

---

## Overrides

### seed_management
N/A — no model training. The only ML model in this system is `text-embedding-004`, a pre-trained Vertex AI service. No random state to fix or log.

### data_snapshot
N/A — no training dataset. Input data is fetched from the live Places API at inference time. The equivalent here is the golden set snapshot (`data/golden/golden_set.json`), which is versioned in git as part of `artifact_versioning()`.

### distribution_monitoring
Partial. Per-request logs record `candidate_count`, `with_reviews`, and `top_scores` on every request — raw data exists. Full distribution monitoring (query length, types distribution, score drift over time) is not aggregated or alerted on. Current gap: no Cloud Monitoring custom metrics or alerting policies for input drift. Implement when traffic volume justifies it.

---

## Owner

Algorithm engineer transitioning to Applied Scientist / MLE at Google. Background: local model design (training/optimization), no prior ML systems, RAG, LLM tuning, or agent experience. **Guide with plain language, explain the "why", and surface insider tips relevant to Google AS interviews and production ML systems.**
