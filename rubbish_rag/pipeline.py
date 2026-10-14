"""RubbishRAG pipeline — the 1-minute version.

Stages (override any of them with @rubbish_rag.* decorators —
see docs/DECORATORS.md for the mechanism):
  chunk -> retrieve (remote) -> rerank -> resolve
"""
from pytest import console_main

import re
import math
import rubbish_rag as R
from .normalize_fa import normalize_fa
from .slots import extract_slots_broken,extract_slots
from collections import defaultdict
from server._hidden_retriever import remote_retrieve



@R.chunker
def chunk(docs):

    MAX_CHARS = 300
    OVERLAP_CHARS = 50
    SEPARATORS = ["\n\n", "\n", " ", ""]

    chunks = []
    for d in docs:
        answer = (d.get("Answer") or d.get("BriefAnswer") or "")
        answer = answer.replace("\r\n", "\n").strip()
        if not answer:
            continue

        question = (d.get("Question") or d.get("Title") or "").strip()
        category = d.get("Category", "")
        header = " | ".join(x for x in (category, question) if x)

        atoms = []
        stack = [(answer, 0, "")]
        while stack:
            text, level, joiner = stack.pop()
            text = text.strip()
            if not text:
                continue
            if len(text) <= MAX_CHARS:
                atoms.append((text, joiner))
                continue

            sep = SEPARATORS[level]
            if sep == "":

                for j in range(0, len(text), MAX_CHARS):
                    atoms.append((text[j:j + MAX_CHARS], joiner if j == 0 else ""))
                continue

            parts = [p for p in text.split(sep) if p.strip()]
            if len(parts) <= 1:

                stack.append((text, level + 1, joiner))
                continue

            for idx in range(len(parts) - 1, -1, -1):
                stack.append((parts[idx], level + 1, joiner if idx == 0 else sep))

        groups, cur, cur_len = [], [], 0
        for text, joiner in atoms:
            add_len = len(text) + (len(joiner) if cur else 0)
            if cur and cur_len + add_len > MAX_CHARS:
                groups.append(cur)

                tail, tl = [], 0
                for a in reversed(cur):
                    l = len(a[0]) + len(a[1])
                    if tl + l > OVERLAP_CHARS:
                        break
                    tail.insert(0, a)
                    tl += l
                cur = tail
                cur_len = (sum(len(t) for t, _ in cur)
                           + sum(len(j) for _, j in cur[1:]))

                if cur and cur_len + len(joiner) + len(text) > MAX_CHARS:
                    cur, cur_len = [], 0
                add_len = len(text) + (len(joiner) if cur else 0)
            cur.append((text, joiner))
            cur_len += add_len
        if cur:
            groups.append(cur)

        for i, group in enumerate(groups):
            body = group[0][0] + "".join(j + t for t, j in group[1:])
            chunks.append({
                "doc_id": d["doc_id"],
                "category": category,
                "question": question,
                "text": f"{header}\n{body}" if header else body,
                "body": body,
                "chunk_id": f'{d["doc_id"]}:{i}',
            })

    print(f"chunk: {len(docs)} docs -> {len(chunks)} chunks")
    return chunks


@R.retriever
def retrieve(query, topk=5):
    """Client-side RRF over query variants + length-compensated rescoring.

    Why: the server fuses with 0.5*bm25/max(bm25) + 0.5*dense/max(dense). One
    outlier in either list shrinks every other normalized score, so ranks 2..k
    become near-ties decided by noise. Ranks are invariant to score scale, so
    RRF = sum 1/(60 + rank) is immune to that. A mild length prior then offsets
    BM25 (k1=1.2, b=0.9), which over-penalizes long, thorough answers.
    """

    print("start retrieve")
    """Client-side RRF over query variants + length-compensated rescoring.

    The server fuses with 0.5*bm25/max(bm25) + 0.5*dense/max(dense), so one
    outlier shrinks every other normalized score and ranks 2..k become
    near-ties. Ranks don't depend on score scale, so RRF = sum 1/(60+rank)
    is immune to that. A mild log length prior then offsets BM25 (k1=1.2,
    b=0.9), which over-penalizes long, thorough answers.
    """
    RRF_K = 60
    POOL_MULT = 5
    LENGTH_ALPHA = 0.15

    depth = max(topk * POOL_MULT, 20)

    norm = normalize_fa(query)
    toks = norm.split(' ')

    keywords = toks
    variants = [(norm, 1.0)]               
    if keywords and len(keywords) < len(toks):
        variants.append((" ".join(keywords), 1.0))
    if norm != query:
        variants.append((query, 0.5))

    fused = defaultdict(float)
    docs = {}
    lens = {}

    for q, weight in variants:
        try:
            res = remote_retrieve(q, topk=depth)
            res = [_item for _item in res if int(_item['doc_id']) <= 9000]
        except Exception:
            continue

        for rank, d in enumerate(res, start=1):

            if isinstance(d, dict):
                did = None
                for key in ("id", "doc_id", "docid", "_id"):
                    if key in d:
                        did = d[key]
                        break
                text = d.get("text") or d.get("content") or ""
                if did is None:
                    did = text
            else:
                did = text = str(d)

            fused[did] += weight / (RRF_K + rank)
            if did not in docs:
                docs[did] = d
                lens[did] = max(len(text.split(' ')), 1)

    if not fused:
        return remote_retrieve(norm, topk=topk)

    sorted_lens = sorted(lens.values())
    median_len = sorted_lens[len(sorted_lens) // 2]

    adjusted = {}
    for did, score in fused.items():

        adjusted[did] = score * (1.0 + LENGTH_ALPHA * math.log1p(lens[did] / median_len))

    ranked = sorted(adjusted, key=adjusted.get, reverse=True)[:topk]

    out = []
    for did in ranked:
        d = docs[did]
        if isinstance(d, dict):
            d = {**d, "score": adjusted[did]}
        out.append(d)

    print(out)
    print("end retrieve")
    print('\n\n')
    return out



@R.reranker
def rerank(query, hits):
    print("start rerank")
    print(query)
    print('****')
    print(hits)
    """Current behavior: orders hits by shared-word count with the query.

    TODO: find a query where the top hit below is clearly the wrong document.
    What fooled the counter? How would you catch that case?
    You decide the ranking rule.
    """

    #####
    query = normalize_fa(query)
    #####

    q = set(query.split())
    for h in hits:
        #####
        h["text"] = normalize_fa(h["text"])
        #####
        toks = set(h["text"].split())
        h["rerank_score"] = len(q & toks)
    res = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)

    print(res)
    print("end rerank")
    print('\n\n')
    return res


@R.resolver
def resolve(query, hits):
    print("start resolve")
    print(query)
    print('******')
    print(hits)
    """Current behavior: always returns an answer built from the top hits.

    TODO: what should happen when the top hits contradict each other, or when
    the query itself is missing the one detail the answer depends on?
    Check the output contract in the README, then decide.
    """
    # _ = extract_slots_broken(query)  # currently unused
    # top = hits[0] if hits else {"text": "", "doc_id": -1}
    # res =  {"status": "answer", "brief": top["text"][:200],
    #         "cites": [h["doc_id"] for h in hits[:3]] or [top["doc_id"]]}


    if not hits:
        return {"status": "no_answer", "brief": "", "cites": []}

    q = extract_slots(query)
    top_hits = hits[:3]
    hit_slots = [extract_slots(h["text"]) for h in top_hits]
    cites = [h["doc_id"] for h in top_hits]
    keys = ("BANK", "RANK", "PERSONA", "DEBT")


    labels = {"BANK": "بانک", "RANK": "رتبه اعتباری", "PERSONA": "حقیقی یا حقوقی بودن"}
    for k in ("BANK", "RANK", "PERSONA"):
        if q[k] is None:
            values = {s[k] for s in hit_slots if s[k] not in (None, "both")}
            if len(values) > 1:
                return {
                    "status": "clarify",
                    "missing": k,
                    "brief": f"لطفاً {labels[k]} را مشخص کنید: " + "، ".join(sorted(values)),
                    "cites": cites,
                }

    mismatches = [
        sum(
            1 for k in keys
            if q[k] and s[k]
            and "both" not in (q[k], s[k])
            and q[k] != s[k]
        )
        for s in hit_slots
    ]
    best = min(range(len(top_hits)), key=lambda i: (mismatches[i], i))

    if mismatches[best] > 0:
        return {"status": "no_answer", "brief": "", "cites": cites}

    for i in range(len(top_hits)):
        if i == best or mismatches[i] > 0:
            continue
        a, b = hit_slots[best], hit_slots[i]
        same_entities = all(a[k] == b[k] for k in keys)
        if same_entities and (a["NEGATED"] != b["NEGATED"] or
                              (a["NUMS"] and b["NUMS"] and set(a["NUMS"]) != set(b["NUMS"]))):
            return {
                "status": "conflict",
                "brief": top_hits[best]["text"][:200],
                "cites": [top_hits[best]["doc_id"], top_hits[i]["doc_id"]],
            }

    top = top_hits[best]
    ordered = [top] + [h for h in top_hits if h is not top]
    res = {
        "status": "answer",
        "brief": top["text"][:200],
        "cites": [h["doc_id"] for h in ordered],
    }


    print(res)
    print("end resolve")
    print('\n\n')

    return res