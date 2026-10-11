"""Persian text normalization for Credit-Scoring QA.

Handles:
  1. Persian / Arabic digits -> ASCII numerals (e.g. ۲۵۰ / ٢٥٠ -> 250)
  2. Arabic character variants -> Canonical Persian (ي -> ی, ك -> ک, ة -> ه)
  3. Half-spaces (ZWNJ \u200c) -> standardized whitespace
  4. Common colloquial speech and enclitic suffixes (رتبم -> رتبه من, میشه -> می‌شود)
  5. Whitespace collapsing and punctuation strip
"""
import re

FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EN_DIGITS = "0123456789"
_DIGIT_MAP = {ord(a): b for a, b in zip(FA_DIGITS + AR_DIGITS, EN_DIGITS * 2)}

COLLOQUIAL_MAP = {
    "میشه": "می‌شود",
    "نمیشه": "نمی‌شود",
    "چیه": "چیست",
    "چقد": "چقدر",
    "رتبم": "رتبه من",
    "رتبه‌م": "رتبه من",
    "امتیازم": "امتیاز من",
    "سابقه‌م": "سابقه من",
    "سابقم": "سابقه من",
    "شرکتم": "شرکت من",
    "چکم": "چک من",
    "وامم": "وام من",
    "قسطام": "اقساط من",
    "قسطامو": "اقساط من را",
    "گزارشم": "گزارش من",
    "گزارشمو": "گزارش من را",
    "مالیاتم": "مالیات من",
    "مالیاتمو": "مالیات من را",
}


def normalize_fa(text: str) -> str:
    """Normalize Persian text to canonical form."""
    if not text:
        return ""
    # 1. Digits mapping
    s = text.translate(_DIGIT_MAP)
    # 2. Arabic letter variants
    s = s.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه").replace("ـ", "")
    # 3. ZWNJ normalization
    s = s.replace("\u200c", " ")
    # 4. Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # 5. Colloquial replacements
    for k, v in COLLOQUIAL_MAP.items():
        s = re.sub(rf"\b{k}\b", v, s)
    return s
