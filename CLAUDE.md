# CLAUDE.md

## Owner
Algorithm engineer transitioning to Applied Scientist / MLE at Google. Background: local model design (training/optimization), no prior ML systems, RAG, LLM tuning, or agent experience. **Guide with plain language, explain the "why", and surface insider tips relevant to Google AS interviews and production ML systems.**

## Production Standards
All code and infrastructure must meet Google production baseline. Apply these by default — do not wait to be asked.

**Reliability**
- Serving must use the same embedding model and version as indexing — a mismatch produces wrong retrieval with no error signal; the most common silent failure in production ML systems (Uber Engineering Blog, "Meet Michelangelo: Uber's Machine Learning Platform," 2017)
- Retry ML inference calls with exponential backoff and jitter, not immediate retry — immediate retries synchronize and amplify load on a struggling service; jitter de-correlates them (AWS Architecture Blog, "Exponential Backoff And Jitter," Marc Brooker, 2015)
- Track p95/p99 serving latency, not averages — LLM inference variance is high; tail latency governs user experience; averages mask the worst 5% of requests (Google Research, Jeff Dean, "The Tail at Scale," CACM 2013)

**Observability**
- Log model inputs, retrieved context, and outputs on every request — HTTP 200 does not mean correct answers; prediction logging is the only way to detect silent quality degradation and data drift in production (Uber Engineering Blog, "Meet Michelangelo," 2017)
- Monitor input feature distributions continuously, not just error rates — distribution shift is the leading cause of silent model degradation; a model can stay "up" for weeks while its inputs drift out of the training domain (Uber Engineering Blog, "Meet Michelangelo," 2017)
- Propagate a unique request_id through all log lines for a given request — without it you cannot reconstruct what happened for a specific failed inference (Google Research, "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure," 2010)

**Evaluation & Experimentation**
- Every model or retrieval change requires offline eval on a golden set before production — "it worked on a few examples" is not evidence; treat offline eval as a mandatory gate, not an optional check (Google, "Rules of ML," Rule #29)
- Shadow evaluation: route candidate models to real traffic, log both result sets, but serve only the current model's answer — measures quality on production queries with zero user risk (Uber Engineering Blog, "Meet Michelangelo," 2017: shadow mode)
- Log a variant ID on every request from the first experiment — without it metric changes cannot be attributed to specific changes; this is the minimum viable A/B infra (Uber Engineering Blog, "Meet Michelangelo," 2017; Netflix Tech Blog, experimentation platform)

**Deployment**
- Version model artifacts, embeddings, and configs together as immutable snapshots — enables instant rollback when quality regresses without retraining; Uber Michelangelo stores versioned model packages that bundle the model, feature config, and hyperparameters (Uber Engineering Blog, "Meet Michelangelo," 2017)
- Use phased rollout (shadow → canary → partial → full) for any model change — never flip 100% of traffic to a new model instantly; measure quality at each stage before proceeding (Uber Engineering Blog, "Meet Michelangelo," 2017; Netflix Tech Blog, ML deployment practices)
- Run a smoke test against the live endpoint after every deploy before marking complete — CI passing proves the image builds, not that the deployed service works; post-deploy synthetic probes are a mandatory rollout gate at Google and top ML teams
