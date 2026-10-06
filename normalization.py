from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

_ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_MARKS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED\u0640]")
_MATCH_MARKS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0640]")
_PUNCTUATION = re.compile(r"[،؛؟!،.,:;'\"()\[\]{}<>«»\-_/\\|@#$%^&*+=~`﴿﴾]")
_SPACES = re.compile(r"\s+")
_ALEF_FORMS = re.compile(r"[أإآٱ]")
_PERSIAN = str.maketrans({"ی": "ي", "ې": "ي", "ک": "ك", "ە": "ه", "ہ": "ه", "ۀ": "ه"})

STOPWORDS = frozenset(
    """من في على ان أن إن الى إلى عن مع ما لا لم لن قد و ثم أو او هو هي هم انت أنتم كان كانت يكون تكون قال قالت
هذا هذه ذلك تلك الذي التي الذين اللاتي اللائي كل بعض غير عند بين حتى إذا اذا لو لكن بل يا أيها ايها""".split()
)


def _prefold(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("ﷲ", "الله")
    return _ZERO_WIDTH.sub("", text).translate(_PERSIAN)


def normalize_strict(text) -> str:
    if not text:
        return ""
    text = _MARKS.sub("", _prefold(text))
    text = _ALEF_FORMS.sub("ا", text)
    return _SPACES.sub(" ", _PUNCTUATION.sub(" ", text)).strip()


def normalize_lenient(text) -> str:
    return normalize_strict(text).replace("ة", "ه").replace("ى", "ي")


def normalize_for_matching(text) -> str:
    if not text:
        return ""
    text = _MATCH_MARKS.sub("", _prefold(text))
    for source, target in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا"), ("ؤ", "و"), ("ئ", "ي"), ("ة", "ه"), ("ى", "ي")):
        text = text.replace(source, target)
    text = re.sub(r"[^\u0621-\u064A\s]", " ", text)   
    return _SPACES.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    return [token for token in text.split() if token]


def content_words(tokens) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def char_ngrams(text: str, n: int = 4) -> set:
    return {text[i : i + n] for i in range(len(text) - n + 1)}


_ARABIC_LETTER = re.compile(r"[\u0621-\u064A]")


def aligned_words(text: str) -> List[Tuple[str, str]]:
    pairs = []
    for word in text.split():
        skeleton = normalize_for_matching(word)
        if skeleton and _ARABIC_LETTER.search(skeleton):
            pairs.append((word, skeleton.replace(" ", "")))
    return pairs


_VOWELS = {"\u064E": "a", "\u064F": "u", "\u0650": "i", "\u064B": "A", "\u064C": "U", "\u064D": "I", "\u0651": "~"}


def vowel_signature(word: str) -> List[Tuple[str, str]]:
    word = _prefold(word).replace("ٱ", "ا")
    signature: List[Tuple[str, str]] = []
    for char in word:
        if char in _VOWELS:
            if signature:
                letter, marks = signature[-1]
                signature[-1] = (letter, "".join(sorted(set(marks + _VOWELS[char]))))
        elif "\u0621" <= char <= "\u064A":
            signature.append((char, ""))
    return signature


_PHONETIC_CLASSES = {"ص": "س", "ث": "س", "ذ": "ز", "ظ": "ز", "ض": "ز", "ط": "ت", "ك": "ق", "ح": "ه", "غ": "ع"}
_PHONETIC_TABLE = str.maketrans(_PHONETIC_CLASSES)


def phonetic_key(skeleton_word: str) -> str:
    word = skeleton_word.translate(_PHONETIC_TABLE)
    return "".join(ch for i, ch in enumerate(word) if i == 0 or ch != word[i - 1])
