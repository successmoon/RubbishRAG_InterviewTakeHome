# Engineering Evidence Report (PROOF.md)

## Candidate Information

- **Candidate Name:** `MAhsa Movaffagh`
- **Submission Date:** `2026-10-14`
- **Execution Target:** `Linux`
- **Track Scope Completed:**
  - [x] Required Core (Tasks 1, 2, 3)
  - [ ] Elective 1: `<Elective Name>`
  - [ ] Elective 2: `<Elective Name>`
  - [ ] Staff Track: Adaptive System-1 Decision Judge

---

## Executive Summary & Scorecard

| Metric | Naive Baseline | Your Final Pipeline | Delta ($\Delta$) | Target |
| :--- | :---: | :---: | :---: | :---: |

accuracy                  | 0.300                | 0.300                | +0.000
answer_acc                | 0.300                | 0.300                | +0.000
clarify_acc               | 0.000                | 0.000                | +0.000
latency_mean_ms           | 17.8                 | 25.3                 | -
latency_p95_ms            | 20.1                 | 37.2                 | -
---



Investigation 1: Persian Digit Normalization & Sparse Retrieval
#### 1. Observed Symptom
Query ۴۸۰ شدم. چرا؟ returns only one non-zero fused hit: doc_id=110 at fused=1.0. Remaining top-5 slots are 0.0, and doc_id=1 is duplicated.

#### 2. Falsifiable Hypothesis
Persian digits (۴۸۰) are not normalized to ASCII (480) before tokenization/BM25 indexing. Documents containing 480 get BM25 term frequency 0, so ranking collapses to dense-only or accidental matches.

#### 3. Diagnostic Probes & Exact Trace IDs

Probe Command: python3 rubbish.py bepar "۴۸۰ شدم. چرا؟"

Baseline Trace ID: trace_1791766181_0755408


#### 5. Root Cause Analysis
No Persian/Arabic digit folding in the baseline normalization pipeline. BM25 cannot match ۴۸۰ to 480.

#### 6. Architectural Modification

Module: using normalization in rubbish_rag/pipeline.py

Mechanism: Map ۰-۹ and ٠-٩ to 0-9 before tokenization for both queries and documents.

#### 7. Ablation Experiment

Disable digit mapping: ۴۸۰ شدم. چرا؟ loses exact BM25 match; top-k becomes dense-dominated.

Specific affected case: ۴۸۰ شدم. چرا؟



#### 1. Observed Symptom
Query رتبه‌م E2 ه، ملی بهم وام می‌ده؟ with topk=350 returns doc_id=229 as top-1, but poison-like IDs 9000 and 9003 appear in top-10, with duplicates and many zero-score tail hits.

#### 2. Falsifiable Hypothesis
ZWNJ (رتبه‌م) and colloquial forms (ه, بهم, می‌ده) are not normalized/standardized. BM25 term frequency drops, allowing dense or poison docs to enter top ranks.

#### 3. Diagnostic Probes & Exact Trace IDs

Probe Command: python3 rubbish.py bepar "رتبه‌م E2 ه، ملی بهم وام می‌ده؟"

Baseline Trace ID: trace_1791766850_945370

#### 5. Root Cause Analysis
No ZWNJ standardization/removal and no colloquial-to-formal mapping. Token mismatch suppresses BM25 for canonical docs.

#### 6. Architectural Modification

Module: using normalization in rubbish_rag/pipeline.py

Mechanism: Normalize ZWNJ, collapse whitespace, optionally map common colloquial forms.



Investigation 3: Credit Grade Code Normalization / Poison Doc Collapse
#### 1. Observed Symptom
رتبه C3 یعنی چی؟ and رتبه C1 یعنی چی؟ both return doc_id=9004 as top-1 with fused=1.0. E2 query also pulls 9000/9003 into top ranks.

#### 2. Falsifiable Hypothesis
Credit grade codes (C1, C3, E2) are not protected as distinct tokens. A poison/placeholder doc (9004) matches all grade queries, causing collapse.

#### 3. Diagnostic Probes & Exact Trace IDs

Probe Command: python3 rubbish.py bepar "رتبه C3 یعنی چی؟"

Trace ID: trace_1791766181_2893376

Probe Command: python3 rubbish.py bepar "رتبه C1 یعنی چی؟"

Trace ID: trace_1791767255_4711623

Probe Command: python3 rubbish.py bepar "رتبه‌م E2 ه، ملی بهم وام می‌ده؟"

Trace ID: trace_1791766850_945370

#### 4. Quantitative Evidence & Sub-score Analysis

Trace ID	Query Variant	Top-1 Doc ID	BM25 Score	Dense Score	Fused Score	Outcome

#### 5. Root Cause Analysis
Grade codes are not tokenized as protected terms, and poison/placeholder docs are not filtered. 9004 dominates multiple grade queries.

#### 6. Architectural Modification

Module: rubbish_rag/pipeline.py

Mechanism: Preserve case for Latin grade codes; add code-aware tokenization; filter poison IDs >=9000.


Investigation 4: Poison / Placeholder Document Contamination
#### 1. Observed Symptom
Poison-like IDs 9000, 9001, 9003, 9004, 9104 appear in top ranks. C3 and C1 both top-1 9004; E2 top-10 includes 9000/9003; عدد صادرات/واردات includes 9000/9003; چرا وقتی شرکت هیچ سابقه تاخیری ندارد... has 9001 at rank 2.

#### 2. Falsifiable Hypothesis
Placeholder/poison docs are indexed and receive high dense or fused scores. No allowlist/denylist or poison filter exists.

#### 3. Diagnostic Probes & Exact Trace IDs

python3 rubbish.py bepar "رتبه C3 یعنی چی؟" → trace_1791766181_2893376

python3 rubbish.py bepar "رتبه C1 یعنی چی؟" → trace_1791767255_4711623

python3 rubbish.py bepar "رتبه‌م E2 ه، ملی بهم وام می‌ده؟" → trace_1791766850_945370

python3 rubbish.py bepar "عدد صادرات/واردات توی رتبه اعتباری موثره؟" → trace_1791766181_2396932

python3 rubbish.py bepar "چرا وقتی شرکت هیچ سابقه تاخیری ندارد، بالاترین امتیاز رو نداره؟" → trace_1791766181_342839


#### 4. Root Cause Analysis
Poison docs are not filtered and can outrank real docs, especially when BM25 is sparse or query uses codes/digits.

#### 5. Architectural Modification

Module: rubbish_rag/pipeline.py

Mechanism: Add denylist/allowlist or poison filter for IDs >=9000; mark as non-answer; exclude from fused ranking.

#### 6. Ablation Experiment

Remove filter: C1 and C3 again return 9004 top-1; E2 returns 9000/9003.

Affected cases: all grade/code and numeric queries above.

#### 7. Production Trade-offs

Latency: Negligible.

Memory: Small ID set.

Edge Cases: Must not exclude legitimate high IDs; make configurable.

Investigation 5: Zero-Score Tail / Top-k Thresholding Failure
#### 1. Observed Symptom
topk=350 for E2 query returns many fused=0.0 hits after useful results. topk=5 for ۴۸۰ شدم. چرا؟ returns zeros in slots 2–5.

#### 2. Falsifiable Hypothesis
No minimum fused-score threshold and no zero-score filtering. Top-k pads results with irrelevant zero-score docs.

#### 3. Diagnostic Probes & Exact Trace IDs

python3 rubbish.py bepar "رتبه‌م E2 ه، ملی بهم وام می‌ده؟" --topk 350 → trace_1791766850_945370

python3 rubbish.py bepar "۴۸۰ شدم. چرا؟" --topk 5 → trace_1791766181_0755408


#### 5. Root Cause Analysis
Pipeline returns full top-k regardless of score. No threshold, no zero filtering, no dedup.

#### 6. Architectural Modification

Module: rubbish_rag/pipeline.py

Mechanism: Add min_fused threshold; filter fused <= 0; cap returned results after dedup.

#### 7. Ablation Experiment

Remove threshold: zero-score tail returns; top-k polluted.

Affected cases: E2 topk=350, ۴۸۰ topk=5.

#### 8. Production Trade-offs

Latency: Negligible.

Memory: Negligible.

Edge Cases: Threshold may remove low-signal but relevant docs; tune per benchmark.



## Reproduction Checklist

Verify that the evaluation committee can reproduce your numbers cleanly:
- [ ] `pytest tests/` passes cleanly (all tests green).
- [ ] `python3 rubbish.py bench` finishes in under 3 minutes on CPU.
- [ ] `python3 rubbish.py plot` generates all diagnostic figures in `plots/`.
- [ ] `python3 rubbish.py bastesh` packages `submissions/submission.zip` with zero errors.