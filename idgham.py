from __future__ import annotations

import re

_MARKER = re.compile(r"^\(\d+\)$")

_SUKUN, _SHADDA, _FATHATAN = "\u0652", "\u0651", "\u064B"
_TANWEEN = set("\u064B\u064C\u064D")
_IDGHAM_AFTER_NOON = set("نمرل")
_IDGHAM_AFTER_LAM = set("لر")
_DIACRITIC_CHARS = set(
    "\u0610\u0611\u0612\u0613\u0614\u0615\u0616\u0617\u0618\u0619\u061A"
    "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670"
    "\u06D6\u06D7\u06D8\u06D9\u06DA\u06DB\u06DC\u06DF\u06E0\u06E1\u06E2\u06E3\u06E4"
    "\u06E7\u06E8\u06EA\u06EB\u06EC\u06ED"
)


def _base_letters(word: str) -> str:
    return "".join(c for c in word if c not in _DIACRITIC_CHARS)


def _insert_shadda(word: str) -> str:
    return word if not word else word[0] + _SHADDA + word[1:]


def _ends_with_tanween(word: str) -> bool:
    if not word:
        return False
    if word[-1] in _TANWEEN:
        return True
    return len(word) >= 2 and word[-1] in "اى" and word[-2] == _FATHATAN


def apply_idgham(text: str, extended: bool = True) -> str:

    words = text.split(" ")
    noon_set = _IDGHAM_AFTER_NOON if extended else set("نم")
    i = 0
    while i < len(words):
        word = words[i]
        if _MARKER.match(word) or not word:
            i += 1
            continue
        j = i + 1
        while j < len(words) and _MARKER.match(words[j]):
            j += 1
        if j < len(words):
            base = _base_letters(words[j])
            first = base[0] if base else ""
            if word.endswith("\u0646" + _SUKUN) and first in noon_set:
                words[i], words[j] = word[:-1], _insert_shadda(words[j])
            elif _ends_with_tanween(word) and first in noon_set:
                words[j] = _insert_shadda(words[j])
            elif word.endswith("\u0645" + _SUKUN) and first == "\u0645":
                words[i], words[j] = word[:-1], _insert_shadda(words[j])
            elif extended and word.endswith("\u0644" + _SUKUN) and first in _IDGHAM_AFTER_LAM:
                words[i], words[j] = word[:-1], _insert_shadda(words[j])
        i += 1
    return " ".join(words)


