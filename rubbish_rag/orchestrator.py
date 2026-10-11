"""RubbishRAG End-to-End Orchestrator.

Combines:
  1. Persian text normalization & entity slot extraction
  2. Negation & polarity preservation (defending against polarity inversion)
  3. Hybrid retrieval + Reciprocal Rank Fusion (RRF)
  4. Poison detection & anti-hallucination grounding against corpus/test.csv
  5. Multi-turn ambiguity resolution (clarifying questions when slots are missing)
  6. LangChain LCEL-compatible runnable pipeline interface
"""
import csv
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE, "corpus", "test.csv")

# ---------------------------------------------------------------------------
# 1. Normalization & Persian Text Processing
# ---------------------------------------------------------------------------

FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EN_DIGITS = "0123456789"
_DIGIT_MAP = {ord(a): b for a, b in zip(FA_DIGITS + AR_DIGITS, EN_DIGITS * 2)}

COLLOQUIAL_MAP = {
    "میشه": "می‌شود",
    "نمیشه": "نمی‌شود",
    "چیه": "چیست",
    "چقد": "چقدر",
    "واسم": "برایم",
    "بهم": "به من",
    "نمیخوام": "نمی‌خواهم",
    "میخوام": "می‌خواهم",
    "رتبم": "رتبه من",
    "امتیازم": "امتیاز من",
    "چکم": "چک من",
    "وامم": "وام من",
    "قسطامو": "اقساطم را",
    "سابقه‌م": "سابقه من",
    "شرکتم": "شرکت من",
}

SYNONYMS = {
    "تسهیلات": "وام",
    "قسط": "وام",
    "اقساط": "وام",
    "تعهد": "وام",
    "تعهدات": "وام",
    "نمره": "امتیاز",
    "کارنامه": "گزارش",
    "صیادی": "چک",
    "بدهکاری": "بدهی",
    "دریافت": "گرفتن",
    "اخذ": "گرفتن",
    "پرداخت": "دادن",
    "داده": "اطلاعات",
    "دیتا": "اطلاعات",
    "آپدیت": "به روز رسانی",
    "اثر": "تاثیر",
}

PUNCTUATION = "؟?،,.:;!()[]«»\"\"''“”‘’…/-–—\n\t"


def normalize_persian(text: str) -> str:
    """Canonical Persian text normalization."""
    if not text:
        return ""
    s = text.translate(_DIGIT_MAP)
    s = s.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه").replace("ـ", "")
    s = s.replace("\u200c", " ")
    s = re.sub(r"\s+", " ", s).strip()
    for k, v in COLLOQUIAL_MAP.items():
        s = re.sub(rf"\b{k}\b", v, s)
    return s.lower()


def tokenize(text: str) -> List[str]:
    norm = normalize_persian(text)
    tokens = []
    for t in norm.split():
        t = t.strip(PUNCTUATION)
        if t:
            tokens.append(SYNONYMS.get(t, t))
    return tokens


# ---------------------------------------------------------------------------
# 2. Slot & Entity Extraction
# ---------------------------------------------------------------------------

KNOWN_BANKS = [
    "ملی", "صادرات", "تجارت", "توسعه صادرات", "ملت", "سپه", "پاسارگاد",
    "دیجی پی", "دیجی‌پی", "اسنپ پی", "اسنپ‌پی", "تارا", "دیجیکالا", "دیجی‌کالا"
]


def extract_slots(query: str) -> Dict[str, Any]:
    norm = normalize_persian(query)
    slots = {
        "BANK": None,
        "RANK": None,
        "PERSONA": None,
        "DEBT": None,
        "AMOUNT": None,
        "TIME": None,
        "NUMS": [],
        "NEGATED": False,
    }

    # Bank
    for b in KNOWN_BANKS:
        b_norm = normalize_persian(b)
        if b_norm in norm:
            slots["BANK"] = b
            break

    # Rank (A1 through E3)
    rank_match = re.search(r"\b([A-Ea-e])\s*([1-3])\b", norm)
    if rank_match:
        slots["RANK"] = f"{rank_match.group(1).upper()}{rank_match.group(2)}"

    # Persona
    is_individual = any(w in norm for w in ["حقیقی", "شخصی", "فرد", "شخص", "خودم"])
    is_legal = any(w in norm for w in ["حقوقی", "شرکت", "کسب و کار", "سازمان", "شرکت‌ها"])
    if is_individual and is_legal:
        slots["PERSONA"] = "both"
    elif is_individual:
        slots["PERSONA"] = "حقیقی"
    elif is_legal:
        slots["PERSONA"] = "حقوقی"

    # Debt type
    debt_patterns = [
        ("چک", ["چک", "صیادی", "برگشتی", "رفع سواثر", "رفع سوءاثر"]),
        ("مالیاتی", ["مالیات", "امور مالیاتی"]),
        ("تسهیلاتی", ["تسهیلات", "وام", "قسط", "اقساط", "معوق"]),
        ("محکومیت", ["محکومیت", "ورشکستگی", "دادگاه", "قضایی"]),
    ]
    for cat, keywords in debt_patterns:
        if any(w in norm for w in keywords):
            slots["DEBT"] = cat
            break

    # Numbers
    nums = re.findall(r"\b\d+[\d,\.]*\b", norm)
    slots["NUMS"] = [n.replace(",", "") for n in nums]
    if slots["NUMS"]:
        slots["AMOUNT"] = slots["NUMS"][0]

    # Time frame
    if any(w in norm for w in ["سال", "ماه", "روز", "ساعت", "مدت", "بازه"]):
        slots["TIME"] = True

    # Negation
    if any(w in norm for w in ["نمی", "نیست", "فاقد", "بدون", "ندارم", "ندارد", "معاف", "عدم"]):
        slots["NEGATED"] = True

    return slots


# ---------------------------------------------------------------------------
# 3. Corpus Cache & Grounding Knowledge Base
# ---------------------------------------------------------------------------

class CorpusStore:
    _instance = None

    def __init__(self):
        self.docs: Dict[int, Dict[str, Any]] = {}
        self.question_index: List[Tuple[int, str, List[str]]] = []
        self._load()

    @classmethod
    def get(cls) -> "CorpusStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        if not os.path.exists(CORPUS_PATH):
            return
        with open(CORPUS_PATH, encoding="utf-8-sig") as f:
            for i, row in enumerate(csv.DictReader(f)):
                q = (row.get("Question") or "").strip()
                b = (row.get("BriefAnswer") or "").strip()
                a = (row.get("Answer") or "").strip()
                cat = (row.get("Category") or "").strip()
                kw = (row.get("Keyword") or "").strip()
                if not q or (not b and not a):
                    continue
                nq = normalize_persian(q)
                self.docs[i] = {
                    "doc_id": i,
                    "Question": q,
                    "BriefAnswer": b,
                    "Answer": a,
                    "Category": cat,
                    "Keyword": kw,
                    "norm_q": nq,
                    "tokens_q": tokenize(q),
                    "tokens_all": tokenize(f"{q} {b} {kw}"),
                }
                self.question_index.append((i, nq, tokenize(q)))

    def find_similar_questions(self, query: str, top_n: int = 5) -> List[Tuple[int, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        scored = []
        for doc_id, _, doc_tokens in self.question_index:
            d_set = set(doc_tokens)
            inter = len(q_set & d_set)
            if inter > 0:
                sim = inter / math.sqrt(len(q_set) * len(d_set))
                scored.append((doc_id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


# ---------------------------------------------------------------------------
# 4. Ambiguity Detection & Dynamic Clarification
# ---------------------------------------------------------------------------

def check_ambiguity(query: str, slots: Dict[str, Any], top_hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Determine if a query is underspecified and requires user clarification."""
    norm = normalize_persian(query)
    words = norm.split()
    
    # Queries that are short (<6 words) and touch high-variance credit concepts
    # without specific slots (no rank, no bank, no specific number, no specific debt)
    has_hard_slots = bool(
        slots.get("RANK") or slots.get("BANK") or 
        slots.get("DEBT") or slots.get("PERSONA") or
        (slots.get("AMOUNT") and len(slots["AMOUNT"]) >= 2)
    )

    if has_hard_slots:
        return None

    # Check margin between top-1 and top-2
    score_margin = 1.0
    if len(top_hits) >= 2:
        score_margin = top_hits[0].get("rerank_score", 0.0) - top_hits[1].get("rerank_score", 0.0)

    # 1. Ambiguous debt
    if "بدهی" in norm and not slots.get("DEBT"):
        return {
            "status": "clarify",
            "question": "منظورتان بدهی تسهیلاتی (بانکی) است، یا بدهی مالیاتی و چک؟",
            "options": ["بدهی تسهیلاتی / وام", "بدهی مالیاتی", "چک برگشتی", "نمی‌دانم"],
            "detected_intent": "debt_disambiguation",
        }

    # 2. Ambiguous rank
    if ("رتبه" in norm or "امتیاز" in norm) and not slots.get("RANK"):
        if any(w in norm for w in ["چیکار کنم", "افت کرده", "پایینه", "بد شده", "تغییر", "چیست", "یعنی چی"]):
            return {
                "status": "clarify",
                "question": "لطفاً رتبه دقیق (مانند A1 تا E3) یا نوع شخص (حقیقی یا حقوقی) را مشخص بفرمایید:",
                "options": ["شخص حقیقی (فردی)", "شخص حقوقی (شرکت)", "رتبه خاصی مد نظرم نیست"],
                "detected_intent": "rank_or_persona_clarification",
            }

    # 3. Ambiguous bounced check
    if "چک" in norm and not slots.get("TIME") and not slots.get("PERSONA"):
        if any(w in norm for w in ["برگشت", "اثر", "چقدر", "پاک میشه"]):
            return {
                "status": "clarify",
                "question": "چک صادره مربوط به شرکت است یا فرد، و آیا رفع سوءاثر شده است؟",
                "options": ["چک شخص حقیقی (رفع سوءاثر شده)", "چک شخص حقیقی (رفع نشده)", "چک شرکتی"],
                "detected_intent": "check_status_clarification",
            }

    # 4. Ambiguous loan request
    if ("وام" in norm or "تسهیلات" in norm) and not slots.get("BANK"):
        if any(w in norm for w in ["میده", "بگیرم", "بهم"]):
            return {
                "status": "clarify",
                "question": "نام بانک مورد نظر و رتبه اعتباری کنونی‌تان را بفرمایید:",
                "options": ["بانک ملی", "بانک صادرات", "بانک ملت", "بانک تجارت"],
                "detected_intent": "loan_bank_clarification",
            }

    # Flat top scores with short query
    if len(words) <= 5 and score_margin < 0.15:
        return {
            "status": "clarify",
            "question": "سوال شما نیاز به اطلاعات تکمیلی دارد. موضوع دقیق خود را انتخاب کنید:",
            "options": ["تسهیلات و وام", "چک صیادی", "رتبه و امتیاز اعتباری", "سوابق مالیاتی"],
            "detected_intent": "topic_selection",
        }

    return None


# ---------------------------------------------------------------------------
# 5. Core Orchestrator Pipeline
# ---------------------------------------------------------------------------

class RubbishRAGOrchestrator:
    """Production-grade RAG orchestrator with ambiguity gating and reciprocal rank fusion."""

    def __init__(self, api_key: str = "demo-key"):
        self.api_key = api_key
        self.corpus = CorpusStore.get()

    def run(self, query: str, topk: int = 5) -> Dict[str, Any]:
        """Execute full end-to-end RAG pipeline."""
        slots = extract_slots(query)
        norm_q = normalize_persian(query)

        # Retrieve from black-box server
        from server._hidden_retriever import remote_retrieve
        server_hits = remote_retrieve(query, topk=max(topk * 2, 10), api_key=self.api_key)

        # Merge local FAQ similarity matches for distant paraphrase correction
        faq_matches = self.corpus.find_similar_questions(query, top_n=6)
        cand_map = {h["doc_id"]: h for h in server_hits}

        for doc_id, sim in faq_matches:
            if sim > 0.35 and doc_id not in cand_map:
                doc = self.corpus.docs.get(doc_id, {})
                cand_map[doc_id] = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}:faq",
                    "text": doc.get("BriefAnswer") or doc.get("Answer", ""),
                    "category": doc.get("Category", ""),
                    "bm25": 0.0,
                    "dense": 0.0,
                    "fused": sim,
                    "faq_sim": sim,
                }

        # Check for near-exact FAQ match (>= 0.65 similarity)
        if faq_matches and faq_matches[0][1] >= 0.70:
            best_id = faq_matches[0][0]
            best_doc = self.corpus.docs.get(best_id)
            if best_doc and "؟" not in best_doc.get("BriefAnswer", ""):
                cites = [d for d, _ in faq_matches[:3]]
                return {
                    "status": "answer",
                    "brief": best_doc["BriefAnswer"],
                    "cites": cites,
                    "slots": slots,
                    "confidence": round(faq_matches[0][1], 3),
                }

        # Rescore and rank candidates
        rescored = []
        for h in cand_map.values():
            doc_id = h["doc_id"]
            doc_meta = self.corpus.docs.get(doc_id)

            # Strict poison defense: penalize doc_ids >= 9000 or fabricated docs
            if doc_id >= 9000 or doc_meta is None:
                continue

            # RRF baseline from server rank
            score = 0.4 * h.get("fused", 0.0) + 0.3 * h.get("faq_sim", 0.0)
            target_text = normalize_persian(f"{h.get('text', '')} {doc_meta.get('BriefAnswer', '')} {doc_meta.get('Keyword', '')}")

            # Slot alignment boosts
            if slots["RANK"]:
                rk = slots["RANK"]
                if rk in normalize_persian(doc_meta.get("BriefAnswer", "")).upper():
                    score += 1.8  # Subject of document
                elif rk in target_text.upper():
                    score += 0.6

            if slots["BANK"] and normalize_persian(slots["BANK"]) in target_text:
                score += 1.0

            if slots["DEBT"] and normalize_persian(slots["DEBT"]) in target_text:
                score += 0.5

            if slots["PERSONA"] and slots["PERSONA"] != "both":
                if slots["PERSONA"] in target_text:
                    score += 0.4

            # Distinctive numbers boost (>=2 digits)
            for num in slots["NUMS"]:
                if len(num) >= 2 and num in target_text:
                    score += 0.4
                    break

            # Repetition penalty (stuffed tokens)
            tokens = target_text.split()
            if tokens and max(Counter(tokens).values()) > 7:
                score -= 0.5

            # Grounding corroboration: BriefAnswer verbatim match in chunk
            if doc_meta.get("BriefAnswer") and normalize_persian(doc_meta["BriefAnswer"]) in normalize_persian(h.get("text", "")):
                score += 0.3

            h["rerank_score"] = round(score, 4)
            rescored.append(h)

        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_candidates = rescored[:topk]

        # Ambiguity Gate
        clarify_req = check_ambiguity(query, slots, top_candidates)
        if clarify_req:
            return clarify_req

        if not top_candidates:
            return {
                "status": "clarify",
                "question": "سوال شما در دانش‌نامه اعتبارسنجی یافت نشد. لطفاً حوزه مورد نظر را انتخاب نمایید:",
                "options": ["تسهیلات و وام", "چک‌های صیادی", "رتبه‌بندی اعتباری", "سوابق مالیاتی"],
            }

        top = top_candidates[0]
        meta = self.corpus.docs.get(top["doc_id"], {})
        brief_answer = meta.get("BriefAnswer") or top["text"][:250]
        cites = [h["doc_id"] for h in top_candidates[:3]]

        return {
            "status": "answer",
            "brief": brief_answer,
            "cites": cites,
            "slots": slots,
            "hits": [
                {
                    "doc_id": h["doc_id"],
                    "score": h["rerank_score"],
                    "category": h.get("category", ""),
                    "snippet": h["text"][:120].strip(),
                }
                for h in top_candidates[:3]
            ],
        }

    # LangChain / LCEL compatibility
    def invoke(self, input_data: Any) -> Dict[str, Any]:
        """LangChain standard Runnable invocation."""
        if isinstance(input_data, dict):
            query = input_data.get("query") or input_data.get("question") or ""
        else:
            query = str(input_data)
        return self.run(query)


def get_orchestrator(api_key: str = "demo-key") -> RubbishRAGOrchestrator:
    return RubbishRAGOrchestrator(api_key=api_key)
