# ML Methods

## Problem Framing

The task is place recommendation given a natural-language query. This is formulated as **retrieval + ranking**, not generation:

1. **Retrieval (recall):** pull a candidate set from a real-world index (Google Places)
2. **Filtering:** apply hard constraints the user stated (rating range)
3. **Ranking:** score candidates by semantic relevance to the user's free-text description

This framing is deliberate. A purely generative approach (asking an LLM to "recommend a ramen shop in SF") hallucinates place names, addresses, and ratings because LLMs have no live access to place databases and do not have ground-truth knowledge of which businesses are currently open. Grounding recall in a live API ensures results are real, current, and geographically accurate.

---

## Stage 1 — LLM Type Suggestion

### What it does

The user states a free-text intent (e.g. "boba shop"). The LLM maps this to a shortlist of valid Google Places API types (e.g. `["cafe", "juice_bar", "tea_house"]`).

### Why use an LLM here

The Places API accepts a fixed vocabulary of ~100 place types. Mapping arbitrary natural language to this vocabulary is a semantic task that a lookup table handles poorly:

- "coffee" → `cafe` (obvious)
- "boba shop" → `cafe`, `juice_bar` (not obvious — no direct "boba" type)
- "izakaya" → `bar`, `restaurant`, `japanese_restaurant` (requires cultural knowledge)
- "speakeasy" → `bar`, `cocktail_bar`, `night_club` (requires inference)

An LLM covers the long tail of inputs with a single call. The alternative — a hand-curated synonym map — would need continuous maintenance and would still miss novel inputs.

### Why constrain the LLM to a fixed type list

The prompt includes the full `_AVAILABLE_TYPES` list. Without this constraint, the LLM would sometimes invent type strings (e.g. `"boba_shop"`) that the Places API silently ignores, producing zero results with no error. Constraining the output vocabulary to the API's actual types prevents this silent failure.

### Model and config

- Model: `gemini-2.5-flash` (low latency, structured output)
- Temperature: `0.1` — near-deterministic. Type mapping is a classification task, not a creative one. Higher temperature adds variance with no quality benefit.
- `response_mime_type: "application/json"` — forces structured JSON output, avoiding markdown fences or prose that would break `json.loads()`.

### What it does NOT do

The LLM is only used for type suggestion in the main pipeline. It does **not** generate result text, does **not** score or re-rank candidates, and does **not** explain why a place was recommended. This keeps latency predictable and results auditable — every output traces back to a Places API response, not a generated string.

---

## Stage 2 — Recall via Nearby Search

### Why proximity-based recall

The Places Nearby Search API returns candidates sorted by proximity within a radius. This is the correct recall strategy for "near me" queries because:

1. **Geographic correctness:** a user asking for "ramen in San Francisco" should not get results from Oakland. Proximity recall bounds results to real geography.
2. **No keyword overfitting:** text-based search APIs return results that match keywords — a place named "Ramen Palace" scores high regardless of whether it is actually good. Proximity recall is keyword-agnostic; ranking happens separately.
3. **Fast and deterministic:** proximity is a geometric computation; the API returns results in <500ms. There is no LLM in the recall path, so recall latency is stable.

### Geocoding

The city + country string is resolved to lat/lng via the Geocoding API, not estimated by the LLM. LLMs hallucinate plausible-looking but wrong coordinates — for example, generating `(37.7749, -122.4194)` for San Francisco is correct, but `(37.76, -122.41)` for "SoMa, SF" may be off by a kilometre and shift the recall circle. A deterministic geocoder is more reliable.

### Field mask and SKU

The Nearby Search request includes a `X-Goog-FieldMask` header that specifies exactly which fields to return. This controls both cost and data shape.

| Field group | Fields | SKU tier | Cost |
|-------------|--------|----------|------|
| Basic | `displayName`, `formattedAddress`, `googleMapsUri`, `primaryType` | Basic | $0.017/req |
| Advanced | `rating`, `userRatingCount`, `priceLevel`, `currentOpeningHours` | Advanced | $0.032/req |
| Preferred | `reviews` | Preferred | $0.040/req |

The `reviews` field was explicitly added despite the cost increase because review text is the primary semantic signal used in Stage 3 re-ranking. Without it, the embedding text falls back to only name + address, which has near-zero signal for atmosphere queries ("quiet", "good for dates", "local hidden gem").

All other fields in the mask are used: `rating`/`priceLevel` in the filter, `currentOpeningHours` in the display, `googleMapsUri` for the output link. No field is fetched speculatively.

### Candidate cap

`max_candidates: 20` — the Nearby Search API hard cap is 20 results per call. It does not support pagination. To get more candidates, the alternative is the Text Search API (which supports `nextPageToken`), noted as a future improvement.

### Radius

Default `1000m` (~10 minute walk). This is intentionally conservative for urban use cases — a wider radius returns more candidates but increases the chance of including places the user cannot practically walk to. The radius is user-configurable via `--radius` in the CLI.

---

## Stage 2.5 — Rating Filter

### What it does

After recall, places are partitioned into:
- `in_range`: rated places with `min_rating ≤ rating ≤ max_rating`
- `unrated`: places with no rating (Google returns `null`)
- Discarded: rated places outside the user's range

The pipeline passes `in_range + unrated` to Stage 3.

### Why keep unrated places

Unrated places are typically new businesses or places with very few reviews. They are not evidence of low quality — they are evidence of sparse data. Discarding them removes potentially excellent new places because the user's constraint ("rating ≥ 4.0") cannot be evaluated, not because the constraint is violated.

This is a recall/precision trade-off: keeping unrated places slightly lowers precision (some will not match the user's preference) but preserves recall for new or under-reviewed candidates. The embedding re-rank then scores them on their available text, so they only appear in results if they are semantically relevant.

---

## Stage 3 — Embedding Re-rank

### What it does

The user's free-text description and all candidate place texts are embedded into a shared vector space. Results are ranked by cosine similarity between the user description vector and each place vector.

### Why embedding similarity, not LLM scoring

LLM-based re-ranking (prompting the model with "rank these 20 places against this description") has two problems at this scale:

1. **Latency:** a single re-rank call with 20 candidates in the prompt takes 2–5 seconds. Embedding 21 texts in one batch call takes ~300ms.
2. **Consistency:** LLMs have variable reasoning about relative rankings. The same prompt with the same candidates can return different orderings across calls (even at temperature 0.0). Cosine similarity is deterministic and directly measures what we care about: semantic alignment between the user's stated preference and the place's text.

LLM re-ranking is implemented (`src/llm.py: rerank_places`) as an alternative and can be swapped in, but is not used in the main pipeline.

### Embedding model

**`text-embedding-004`** (768 dimensions, Google). Chosen because:
- Same model family as the rest of the Vertex AI stack — no cross-provider authentication
- 768 dims is a good balance of quality vs compute cost for semantic similarity
- The `task_type` defaults to `RETRIEVAL_DOCUMENT` — appropriate for asymmetric retrieval (user query vs. place document)

**Critical constraint:** the embedding model must not change without re-embedding all place texts. A user description embedded with model version A and a place text embedded with model version B have incommensurable vector spaces — cosine similarity will be meaningless with no error signal. This constraint is encoded in `config.yaml` as a comment and enforced by pinning the model string in config rather than hardcoding it in each call site.

### Place text construction (`build_place_text`)

```python
def build_place_text(p: Place) -> str:
    parts = [p.primary_type or "", p.name, p.address]
    parts.extend(p.reviews)   # up to 5 snippets
    return ". ".join(filter(None, parts))
```

The text concatenates signals in order of increasing semantic richness:
1. `primary_type` — coarse category signal ("ramen_restaurant")
2. `name` — brand/concept signal ("Mensho Tokyo")
3. `address` — neighbourhood signal ("Inner Richmond, SF")
4. `reviews` — atmosphere and quality signal ("rich tonkotsu broth", "always packed on weekends")

Review text carries the most signal for vibe/atmosphere queries. Name and type carry signal when reviews are absent (new places). The fallback is graceful: no reviews still produces a useful embedding; it just ranks on category and geography rather than atmosphere.

**Why up to 5 reviews:** the Places API returns at most 5 review snippets per place in the Preferred SKU. Concatenating all available reviews maximises semantic coverage per place without extra API calls.

### Batching

All texts — the user description plus all `N` place texts — are embedded in a **single API call**:

```python
all_texts = [description] + place_texts   # 1 + N texts
vectors = embed_texts(all_texts, request_id)
user_vec = vectors[0]
place_vecs = vectors[1:]
```

`text-embedding-004` charges per character, not per API call. Batching has no cost penalty and reduces latency from `O(N)` sequential calls to `O(1)`. With `N=20` candidates, this is a 20× latency reduction compared to embedding each text individually.

### Cosine similarity

Pure Python implementation (no numpy). Cosine measures the angle between two vectors, ignoring their magnitude:

```
similarity(a, b) = dot(a, b) / (|a| × |b|)
```

Magnitude normalisation is appropriate here because embedding models already normalise their output vectors to unit length. Using dot product alone would also work (and is faster), but cosine is correct even if the model changes to one that does not normalise.

A score of 1.0 means identical direction (perfect semantic match). In practice, scores for strong matches are in the 0.7–0.9 range; a score below 0.5 indicates low relevance.

---

## Variant ID and Experiment Tracking

Every request log carries `variant_id` (set in `config.yaml: serving.variant_id`). This is the minimum viable A/B infrastructure:

- When the pipeline changes (new embedding model, new prompt, new field mask), bump `variant_id` in config and redeploy.
- Every log line from that point carries the new ID.
- To compare quality before and after a change, filter logs by `variant_id` — no need for a separate experiment tracking system at this scale.

Without `variant_id`, a metric change in Cloud Monitoring cannot be attributed to a specific code change. Even if only one deploy happened, the variant ID confirms it.

---

## What the Pipeline Does Not Do (and Why)

### No LLM re-ranking in production

The LLM re-ranker (`rerank_places`) is implemented but not used in the main pipeline. The current embedding re-rank is faster, cheaper, and more consistent. The LLM re-ranker would be useful if the user's query involves complex constraint satisfaction ("I want a place that's lively on Friday nights but quiet for working on weekday mornings") that cosine similarity cannot capture. This is a known gap, not an oversight.

### No photo embeddings

Google Places returns photo references (not image bytes). Using visual signal would require:
1. Fetching each photo as a raw image (extra API call per photo)
2. Running Gemini Vision to generate a text description
3. Appending that text to the place text before embedding

This adds ~500ms per place and cost per image. The current text-only embedding already provides strong signal from review text. Photo embeddings are a candidate differentiation if text signal proves insufficient for atmosphere-heavy queries — noted as a TODO in `src/places.py`.

### No online learning

The ranking is frozen at training/design time (the embedding model weights). There is no feedback loop from user clicks or explicit ratings back into the model. This is appropriate at the current scale — online learning requires a feedback collection mechanism, a training pipeline, and safeguards against feedback loops (e.g. popular places getting more clicks → higher rank → more clicks). The prediction logs provide the raw data to build this if needed.

---

## Known Limitations

| Limitation | Impact | Mitigation path |
|------------|--------|-----------------|
| Nearby Search cap at 20 candidates | Sparse recall for rare types in large cities | Switch to Text Search API with pagination |
| No user feedback loop | Ranking cannot improve from real usage | Log click/selection events; build a golden set for offline eval |
| Reviews may be in any language | Multilingual text in the embedding; cosine may be unreliable across languages | Filter to English reviews or use a multilingual model |
| Unrated places always included | May lower precision for users who want verified quality | Make this configurable per request |
| Single radius, no density adaptation | Urban (1km) and suburban (5km) need different defaults | Detect city density from geocode response and adapt |
