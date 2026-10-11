# Engineering Evidence Report (PROOF.md)

> **Candidate Note:** Complete this report to document your investigation, architectural decisions, and empirical results. Ground every finding in exact trace IDs from `traces/trace.jsonl` and generated diagnostic plots. Keep it concise, rigorous, and technical.

---

## Candidate Information

- **Candidate Name:** `<Your Name>`
- **Submission Date:** `YYYY-MM-DD`
- **Execution Target:** `macOS / Linux / WSL (CPU)`
- **Track Scope Completed:**
  - [x] Required Core (Tasks 1, 2, 3)
  - [ ] Elective 1: `<Elective Name>`
  - [ ] Elective 2: `<Elective Name>`
  - [ ] Staff Track: Adaptive System-1 Decision Judge

---

## Executive Summary & Scorecard

Summarize your quantitative improvements over the naive baseline. Run `python3 rubbish.py compare` to extract these numbers.

| Metric | Naive Baseline | Your Final Pipeline | Delta ($\Delta$) | Target |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | ~28.6% | `__._%` | `+__.__%` | $\ge 65\%$ |
| **Retrieval Recall@5** | ~42.0% | `__._%` | `+__.__%` | $\ge 75\%$ |
| **Clarification Accuracy** | 0.0% | `__._%` | `+__.__%` | $\ge 50\%$ |
| **Mean Query Latency (CPU)**| ~12 ms | `__._ ms` | `+__._ ms` | $< 100$ ms |
| **Distractor / Poison Ingestion**| ~35.0% | `0.0%` | `-35.0%` | $0.0\%$ |

---

## Investigation & Architectural Changes

*(Duplicate this section for each significant issue investigated across Core & Electives. Aim for 3–5 well-grounded sections.)*

### Investigation 1: `<e.g. Persian Character Normalization & Digit Drift>`

#### 1. Observed Symptom
*Describe the failure mode observed during initial baseline exploration or benchmark runs. Which queries or document types failed?*
> Example: Queries containing Arabic `ي` or Persian digits `۳` failed to match exact credit scoring rules in doc #42 despite high semantic relevance.

#### 2. Falsifiable Hypothesis
*What specific mechanism do you hypothesize is causing this failure?*
> Example: Tokenization and character mismatch between Arabic `ي/ك` vs Persian `ی/ک`, and ASCII digits vs Persian/Arabic digits, causes BM25 term frequency to drop to 0.

#### 3. Diagnostic Probes & Exact Trace IDs
*List the exact CLI probe commands executed and trace IDs logged in `traces/trace.jsonl` (inspect with `python3 rubbish.py inspect-trace <id>`).*
- **Probe Command:** `python3 rubbish.py bepar "رتبه اعتباری ۳ چیست؟"`
- **Baseline Trace ID:** `trace_20261007_140211_0`
- **Follow-up Probe Command:** `python3 rubbish.py bepar "رتبه اعتباری 3 چیست؟"`
- **Normalized Trace ID:** `trace_20261007_140522_1`

#### 4. Quantitative Evidence & Sub-score Analysis
*Compare raw BM25, dense similarity, fused score, and rank position before and after.*

| Trace ID | Query Variant | Top-1 Doc ID | BM25 Score | Dense Score | Fused Score | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `trace_..._0` | Pers. Digits (`۳`) | 9002 (Poison) | 0.000 | 0.512 | 0.256 | Miss |
| `trace_..._1` | Normalized (`3`) | 42 (Target) | 8.421 | 0.835 | 0.814 | **Hit** |

#### 5. Root Cause Analysis
*Explain the technical root cause in the baseline code.*
> Explanation of exact baseline deficiency...

#### 6. Architectural Modification
*What code change or module override was implemented? (Reference file and function).*
- **Module:** `rubbish_rag/normalize_fa.py` & `rubbish_rag/pipeline.py`
- **Mechanism:** Implemented zero-dependency Persian normalizer handling Unicode normalization, Arabic-to-Persian char folding, Persian/Arabic digit conversion to ASCII, and ZWNJ standardization.

#### 7. Ablation Experiment
*What happens if this specific fix is disabled while keeping all other fixes enabled?*
- Score drop when removed: Overall accuracy drops from `XX%` to `YY%`.
- Specific test cases affected: ...

#### 8. Production Trade-offs
- **Latency Overhead:** `< 0.2 ms` per query string on CPU.
- **Memory Overhead:** Negligible (pure regex/table lookup).
- **Edge Cases Considered:** Preservation of Latin brand names (e.g., SWIFT, IBAN), proper handling of floating-point Persian decimal separators.

---

### Investigation 2: `<e.g. Entity Sensitivity & Slot-Aware Reranking>`

#### 1. Observed Symptom
...
#### 2. Falsifiable Hypothesis
...
#### 3. Diagnostic Probes & Exact Trace IDs
- **Probe Commands:** `python3 rubbish.py bepar "..."`
- **Trace IDs:** `trace_...`
#### 4. Quantitative Evidence & Sub-score Analysis
...
#### 5. Root Cause Analysis
...
#### 6. Architectural Modification
...
#### 7. Ablation Experiment
...
#### 8. Production Trade-offs
...

---

### Investigation 3: `<e.g. Ambiguity Detection & Clarification State Machine>`

#### 1. Observed Symptom
...
#### 2. Falsifiable Hypothesis
...
#### 3. Diagnostic Probes & Exact Trace IDs
- **Probe Commands:** `python3 rubbish.py bepar "بدهی دارم"`
- **Trace IDs:** `trace_...`
#### 4. Quantitative Evidence & Sub-score Analysis
...
#### 5. Root Cause Analysis
...
#### 6. Architectural Modification
...
#### 7. Ablation Experiment
...
#### 8. Production Trade-offs
...

---

## Deliberate Non-Interventions

*What potential changes or complex solutions did you evaluate but **deliberately decide NOT to implement**? Explain your engineering rationale (e.g. latency budget, overfitting risk, unnecessary external dependency).*

1. **Rejected: Heavy 7B Parameter Bi-Encoder Embedding Model**
   - *Rationale:* Would increase query latency from ~15ms to >600ms on CPU and require gigabytes of disk/RAM, violating local developer laptop feasibility.
2. **Rejected: Blindly Increasing Top-K to 20**
   - *Rationale:* Degraded precision and increased distractor ingestion rate by 40% without meaningful recall gains.
3. **Rejected: Hardcoded String Matching for Exact Questions**
   - *Rationale:* Would overfit to visible benchmark queries and fail generalization on the sealed evaluation suite.

---

## Bonus Track: Adaptive System-1 Decision Judge *(Optional)*

*If completing the Staff Track, summarize your judge implementation here.*

- **Judge Architecture:** Heuristic Rule-Based / Open-Weight SLM (Llama-3.2 / Qwen)
- **Routing Accuracy on `data/judge_dataset.json`:** `__._%`
- **Clarification F1:** `_.___`
- **Token / Latency Savings:** `__._%`
- **Probability Calibration:**
  - Brier Score: `0.0___`
  - Expected Calibration Error (ECE): `0.0___`
- **Key Insight:** *Explain how your dynamic stopping policy decides when retrieved evidence is sufficient to answer.*

---

## Reproduction Checklist

Verify that the evaluation committee can reproduce your numbers cleanly:
- [ ] `pytest tests/` passes cleanly (all tests green).
- [ ] `python3 rubbish.py bench` finishes in under 3 minutes on CPU.
- [ ] `python3 rubbish.py plot` generates all diagnostic figures in `plots/`.
- [ ] `python3 rubbish.py bastesh` packages `submissions/submission.zip` with zero errors.