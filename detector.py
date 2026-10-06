from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from normalization import normalize_for_matching, normalize_lenient, normalize_strict, tokenize


@dataclass
class DetectedSpan:
    start: int
    end: int                        
    label: str                      
    confidence: Optional[float]     
    source: str                     
    text: str = ""
    hint: Optional[str] = None      


QUOTE_CHARS = " \t\r\n\"“”«»﴿﴾{}()[]"


def trim_span(text: str, start: int, end: int):
    while start < end and text[start] in QUOTE_CHARS:
        start += 1
    while end > start and text[end - 1] in QUOTE_CHARS + ".،,؛:":
        end -= 1
    return start, end


AYAH_TRIGGERS = [
    "قال الله", "قوله تعالى", "قال تعالى", "يقول الله", "يقول تعالى", "قال سبحانه", "قوله سبحانه", "في كتابه",
    "سورة", "الآية", "الاية", "الآيات", "القرآن", "القران", "كتاب الله", "عز وجل", "جل جلاله", "فقال تعالى",
    "ذكر الله", "﴿",
]
HADITH_TRIGGERS = [
    "رسول الله", "النبي", "صلى الله عليه وسلم", "ﷺ", "عليه الصلاة والسلام", "حديث", "رواه", "روى", "متفق عليه",
    "الحديث", "فقال", "قال ص", "صلى الله عليه", "وسلم",
]
_FORMULA_WORDS = {
    normalize_for_matching(w)
    for w in "قال قالت رسول الله صلى عليه وسلم النبي تعالى سبحانه عز وجل فقال يقول الكريم الشريف الحديث الآية روى رواه عن أن أنه البخاري ومسلم".split()
}
_BRACKET_PAIRS = [("“", "”"), ("«", "»"), ("﴿", "﴾"), ("{", "}"), ("(", ")"), ("[", "]")]


class RuleDetector:

    def __init__(self, retriever=None, min_words: int = 3, context_chars: int = 110,
                 min_corpus_cov: float = 0.6, decouple_triggers: bool = True, quoted_min_cov: float = 0.5,
                 quoted_min_cov_hadith: float = 0.75) -> None:
       
        self.kb, self.min_words, self.context_chars, self.min_corpus_cov = retriever, min_words, context_chars, min_corpus_cov
        self.decouple_triggers = decouple_triggers and retriever is not None
        self.quoted_min_cov, self.quoted_min_cov_hadith = quoted_min_cov, quoted_min_cov_hadith

    @staticmethod
    def _segments(text: str):
        segments = []
        quote_positions = [m.start() for m in re.finditer('"', text)]
        if len(quote_positions) % 2 == 0:
            pairs = zip(quote_positions[0::2], quote_positions[1::2])   
        else:   
            pairs = zip(quote_positions, quote_positions[1:])
        for a, b in pairs:
            segments.append((a + 1, b))
        for opener, closer in _BRACKET_PAIRS:
            for m in re.finditer(re.escape(opener) + r"(.*?)" + re.escape(closer), text, re.S):
                segments.append((m.start(1), m.end(1)))
        return segments

    @staticmethod
    def _trigger_type(context: str) -> Optional[str]:
        best_end, best_label = -1, None
        for label, triggers in (("Ayah", AYAH_TRIGGERS), ("Hadith", HADITH_TRIGGERS)):
            for trigger in triggers:
                pos = context.rfind(trigger)
                if pos >= 0 and pos + len(trigger) > best_end:
                    best_end, best_label = pos + len(trigger), label
        return best_label

    def _corpus_coverage(self, span: str):
        if self.kb is None:
            return 0.0, 0.0
        quran_words = set(tokenize(normalize_strict(span)))
        if not quran_words:
            return 0.0, 0.0
        quran_cov = self._quran_window_coverage(quran_words, span)
        hadith_words = set(tokenize(normalize_lenient(span)))
        hadith_cov = max(
            (len(hadith_words & set(tokenize(normalize_lenient(c["text"])))) / len(hadith_words)
             for c in self.kb.search_hadith(span, top_k=5)),
            default=0.0,
        ) if hadith_words else 0.0
        return quran_cov, hadith_cov

    def _quran_window_coverage(self, quran_words: set, span: str) -> float:
        kb, best = self.kb, 0.0
        for cand in kb.search_quran_ayahs(span, top_k=5):
            surah = kb.quran_by_surah.get(cand["surah_id"], {})
            for first in range(cand["ayah_id"] - 2, cand["ayah_id"] + 1):
                words: set = set()
                for length in (1, 2, 3):
                    idx = surah.get(first + length - 1)
                    if idx is None:
                        break
                    ayah_words = set(tokenize(normalize_strict(kb.quran[idx]["text"])))
                    if length > 1 and not any(len(w) >= 4 for w in quran_words & ayah_words):
                        break   
                    words |= ayah_words
                    if first <= cand["ayah_id"] <= first + length - 1:
                        best = max(best, len(quran_words & words) / len(quran_words))
        return best

    def _label_from_corpus(self, n_words: int, trigger: Optional[str], quran_cov: float, hadith_cov: float) -> Optional[str]:
        quran_ok = quran_cov >= self.quoted_min_cov
        hadith_ok = hadith_cov >= self.quoted_min_cov_hadith   
        if (quran_ok or hadith_ok) and n_words >= 4:
            if trigger and ((quran_ok if trigger == "Ayah" else hadith_ok)) and abs(quran_cov - hadith_cov) < 0.25:
                return trigger
            if quran_ok and hadith_ok:
                return "Ayah" if quran_cov >= hadith_cov else "Hadith"
            return "Ayah" if quran_ok else "Hadith"
        return trigger   

    def detect(self, text: str) -> List[DetectedSpan]:
        candidates = []
        for start, end in self._segments(text):
            start, end = trim_span(text, start, end)
            if end <= start:
                continue
            inner = text[start:end]
            words = [w for w in normalize_for_matching(inner).split() if w]
            if len(words) < 2 or len(inner) > 3000:
                continue
            trigger = self._trigger_type(text[max(0, start - self.context_chars):start])
            if len(words) < self.min_words and not trigger:   
                continue
            if sum(w in _FORMULA_WORDS for w in words) / len(words) >= 0.6:
                continue
            quran_cov, hadith_cov = self._corpus_coverage(inner)
            label = None
            if self.decouple_triggers:
                label = self._label_from_corpus(len(words), trigger, quran_cov, hadith_cov)
            elif trigger:
                label = trigger
                other, mine = (hadith_cov, quran_cov) if trigger == "Ayah" else (quran_cov, hadith_cov)
                if other >= 0.8 and mine < 0.5:
                    label = "Hadith" if trigger == "Ayah" else "Ayah"
            elif max(quran_cov, hadith_cov) >= self.min_corpus_cov and len(words) >= 4:
                label = "Ayah" if quran_cov >= hadith_cov else "Hadith"
            if label is None:
                continue
            candidates.append((bool(trigger), max(quran_cov, hadith_cov), end - start, start, end, label, trigger))

        candidates.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)   
        taken = []
        for _, _, _, start, end, label, hint in candidates:
            if all(end <= t_start or start >= t_end for t_start, t_end, _, _ in taken):
                taken.append((start, end, label, hint))
        taken.sort()
        return [DetectedSpan(s, e, label, None, "rules", text[s:e], hint) for s, e, label, hint in taken]
