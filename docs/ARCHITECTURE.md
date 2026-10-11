# ARCHITECTURE — components, data flow, and current behavior

> This document describes what the system **currently does**. What it
> *should* do is your investigation. Discovering and verifying the appropriate
> fixes through experimental evidence is the core of this challenge.

```
corpus/test.csv
  Question | Category | BriefAnswer | Answer | Keyword   (~330 usable rows)
       │
       ▼  rubbish_rag/pipeline.py :: chunk
chunks: {doc_id, chunk_id, text, category}      300-char slices of Answer
       │
       ▼  POST /retrieve  (server/app.py → sealed scorer)
hits:   {doc_id, chunk_id, text, category, bm25, dense, fused}   top-k by fused
       │
       ▼  rubbish_rag/pipeline.py :: rerank  (shared-word count, descending)
       ▼  rubbish_rag/pipeline.py :: resolve (answers from top hits)
{"status":"answer","brief","cites":[doc_ids]}  |  {"status":"clarify","question","options"}
```

## 1. Components (present tense)

- **Corpus** (`corpus/test.csv`). Persian credit-scoring QA. Observe: formal
  vs colloquial phrasing (`می‌شود` / `میشه`), mixed Persian/Latin digits
  (`۲۵۰` / `250`), Latin codes inside Persian sentences (`A1..E3`), amounts
  with units, dates and bank names. Rows in the same `Category` often share
  most of their vocabulary.
- **Chunker** (`@R.chunker`). Input: docs. Output: `{doc_id, chunk_id, text,
  category}`. Current: fixed 300-char slices of `Answer`.
  TODO: inspect what a slice carries and what it drops. Can a slice be judged
  alone? Decide the strategy yourself.
- **Retriever** (`@R.retriever`). Input: raw query. Output: top-k hits, each
  with `bm25`, `dense`, `fused`. Current: forwards the query untouched.
  TODO: map the score fields first — how do `bm25`, `dense` and `fused` relate
  across 50+ logged probes? What varies with query length, document length,
  repeated words, paraphrase?
- **Reranker** (`@R.reranker`). Current: shared-word count.
  TODO: construct queries where the count orders documents the way a human
  would not. Characterise them.
- **Resolver** (`@R.resolver`). Current: always `answer` from top hits.
  TODO: find queries for which *no* hit deserves an answer, and queries whose
  correct answer changes with one swapped word. What should the output be in
  each case? The contract allows `clarify` — when is it the right call, and
  what must it contain to be gradeable?
- **CLI + traces** (`rubbish.py`, `traces/`). `bepar` shows per-hit scores;
  `bench` aggregates the visible 20; `bekesh` plots score spread, length
  effects and hit quality from *your* logs; `bastesh` packs the submission.
  TODO: every claim in your PROOF must cite trace entries. Design probes that
  isolate one variable at a time.

## 2. Sealed surface (honor code)

`server/` internals, hidden queries and gold answers are off-limits for
reading. Your instruments are: `/retrieve` (+ per-hit scores), `/visible`,
`/submit` (aggregates only), quota/usage endpoints, and the trace log.
Findings must be reproducible from logged calls.

## 3. Output contract (required for grading)

```json
{"status": "answer",  "brief": "<one line>", "cites": [doc_id, ...]}
{"status": "clarify", "question": "<...؟>",   "options": ["...", "..."]}
```

Retrieval is scored hit@3 (gold in first three cites); clarification items are
scored on `status`. Numbers, codes and names that the question depends on must
survive into the answer — check this yourself on the visible bench before
submitting.

## 4. Reading pointers (after you have data, not before)

- Cormack, Clarke & Buettcher, SIGIR 2009 — rank fusion vs score fusion.
- Lv & Zhai, SIGIR 2011 — BM25 length normalisation and its fixes.
- Formal et al., 2021 (SPLADE) — learned sparse expansion.
- Turnbull & Berryman, *Relevant Search* — keyword dilution, retrieval tuning.
- Huyen, *AI Engineering* (2025) — production RAG, context/latency discipline.
- Asai et al., 2023 (Self-RAG) — retrieve → critique → generate.
- Aliannejadi et al., SIGIR 2019 — when to ask clarifying questions.
- Es et al., 2023 (RAGAS) — faithfulness / precision style metrics.

The list is deliberately unmapped: your plots decide which of these (if any)
apply to this box.
