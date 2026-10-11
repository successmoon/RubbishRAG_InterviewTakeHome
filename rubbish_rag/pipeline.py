"""RubbishRAG pipeline — the 1-minute version.

Stages (override any of them with @rubbish_rag.* decorators —
see docs/DECORATORS.md for the mechanism):
  chunk -> retrieve (remote) -> rerank -> resolve
"""
import rubbish_rag as R
from .normalize_fa import normalize_fa
from .slots import extract_slots_broken


@R.chunker
def chunk(docs):
    """Current behavior: splits Answer into fixed 300-char pieces.

    TODO: read a few long Answers, then look at what comes out here.
    Can every chunk below be judged on its own? What is missing?
    You decide the chunking strategy.
    """
    chunks = []
    for d in docs:
        text = (d.get("Answer") or d.get("BriefAnswer") or "")
        for i in range(0, len(text), 300):
            part = text[i:i + 300]
            if part.strip():
                chunks.append({
                    "doc_id": d["doc_id"],
                    "category": d.get("Category", ""),
                    "text": part,
                    "chunk_id": f'{d["doc_id"]}:{i // 300}',
                })
    return chunks


@R.retriever
def retrieve(query, topk=5):
    """Current behavior: forwards the raw query to the remote API untouched.

    TODO: what happens to your query on the way there, and to the scores
    on the way back? Probe it (rubbish bepar) before changing anything.
    You may transform the query, the call, and/or the hits.
    """
    from server._hidden_retriever import remote_retrieve
    return remote_retrieve(query, topk=topk)


@R.reranker
def rerank(query, hits):
    """Current behavior: orders hits by shared-word count with the query.

    TODO: find a query where the top hit below is clearly the wrong document.
    What fooled the counter? How would you catch that case?
    You decide the ranking rule.
    """
    q = set(query.split())
    for h in hits:
        toks = set(h["text"].split())
        h["rerank_score"] = len(q & toks)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)


@R.resolver
def resolve(query, hits):
    """Current behavior: always returns an answer built from the top hits.

    TODO: what should happen when the top hits contradict each other, or when
    the query itself is missing the one detail the answer depends on?
    Check the output contract in the README, then decide.
    """
    _ = extract_slots_broken(query)  # currently unused
    top = hits[0] if hits else {"text": "", "doc_id": -1}
    return {"status": "answer", "brief": top["text"][:200],
            "cites": [h["doc_id"] for h in hits[:3]] or [top["doc_id"]]}
