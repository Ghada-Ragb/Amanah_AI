from __future__ import annotations

from difflib import SequenceMatcher
from math import ceil
from typing import Container, Dict, List, Optional, Sequence, Tuple

from idgham import apply_idgham
from normalization import aligned_words, vowel_signature


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    if m < n:
        a, b, m, n = b, a, n, m
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            curr[j + 1] = prev[j] + 1 if a[i] == b[j] else max(curr[j], prev[j + 1])
        prev = curr
    return prev[n]


def allowed_gap(n_tokens: int) -> int:
    return 0 if n_tokens <= 3 else max(1, ceil(0.2 * n_tokens))


def find_window(quote: List[str], source: List[str], gap: int) -> Tuple[int, int]:
    n, window = len(quote), len(quote) + gap
    if len(source) <= window + 2 * gap + 8:
        return 0, len(source)
    quote_set = set(quote)
    prefix = [0]
    for word in source:
        prefix.append(prefix[-1] + (word in quote_set))
    starts = sorted(range(len(source) - window + 1), key=lambda i: prefix[i + window] - prefix[i], reverse=True)[:8]
    best_start = max(starts, key=lambda i: lcs_length(quote, source[i : i + window]))
    return max(0, best_start - gap), min(len(source), best_start + window + gap)


def contains_sequence(source: Sequence[str], quote: Sequence[str]) -> bool:
    n = len(quote)
    return n > 0 and any(source[i : i + n] == list(quote) for i in range(len(source) - n + 1))


def locate_region(q_norm: List[str], s_norm: List[str], gap: int, slack: Optional[int] = None):
    slack = gap if slack is None else slack
    lo, hi = find_window(q_norm, s_norm, gap)
    matcher = SequenceMatcher(None, q_norm, s_norm[lo:hi], autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if blocks:
        first, last = blocks[0], blocks[-1]
        base = lo
        lo = base + max(0, first.b - first.a - slack)
        hi = min(hi, base + last.b + last.size + (len(q_norm) - last.a - last.size) + slack)
        matcher = SequenceMatcher(None, q_norm, s_norm[lo:hi], autojunk=False)
    return lo, hi, matcher


def best_region(quote: str, source: str) -> str:

    q_pairs, s_pairs = aligned_words(quote), aligned_words(source)
    if not q_pairs or len(s_pairs) <= len(q_pairs) + allowed_gap(len(q_pairs)) + 2:
        return source
    q_norm, s_norm = [p[1] for p in q_pairs], [p[1] for p in s_pairs]
    lo, hi, _ = locate_region(q_norm, s_norm, allowed_gap(len(q_norm)), slack=0)
    return " ".join(p[0] for p in s_pairs[lo:hi]) if hi > lo else source


def _diacritic_notes(quote_words: List[str], source_words: List[str], idgham_words: List[str]) -> List[dict]:
    notes = []
    for q_word, s_word, g_word in zip(quote_words, source_words, idgham_words):
        q_sig = vowel_signature(q_word)
        if not any(marks for _, marks in q_sig):
            continue
        conflict = True
        for variant in (s_word, g_word):
            v_sig = vowel_signature(variant)
            if len(v_sig) == len(q_sig) and all(set(qm) <= set(vm) for (_, qm), (_, vm) in zip(q_sig, v_sig)):
                conflict = False
                break
        if conflict:
            notes.append({"word": q_word, "source_word": s_word})
    return notes


def _orthographic_variant(q_tokens: Sequence[str], s_tokens: Sequence[str], vocabulary: Optional[Container[str]]) -> bool:

    if "".join(q_tokens) == "".join(s_tokens):
        return True
    if vocabulary is not None and len(q_tokens) == 1 and len(s_tokens) == 1:
        q, s = q_tokens[0], s_tokens[0]
        return q.replace("ا", "") == s.replace("ا", "") and q not in vocabulary
    return False


def align(quote: str, source: str, vocabulary: Optional[Container[str]] = None) -> Dict[str, object]:

    q_pairs, s_pairs = aligned_words(quote), aligned_words(source)
    idgham_pairs = aligned_words(apply_idgham(source)) if len(source) < 20000 else s_pairs
    if len(idgham_pairs) != len(s_pairs):
        idgham_pairs = s_pairs
    q_orig, q_norm = [p[0] for p in q_pairs], [p[1] for p in q_pairs]
    s_orig, s_norm = [p[0] for p in s_pairs], [p[1] for p in s_pairs]
    g_orig = [p[0] for p in idgham_pairs]

    gap = allowed_gap(len(q_norm))
    lo, hi, matcher = locate_region(q_norm, s_norm, gap)

    raw_ops = [(tag, i1, i2, j1 + lo, j2 + lo) for tag, i1, i2, j1, j2 in matcher.get_opcodes()]
    
    deleted = {w for tag, i1, i2, _, _ in raw_ops if tag in ("delete", "replace") for w in q_norm[i1:i2]}
    for position in (0, -1):
        if len(raw_ops) > 1 and raw_ops[position][0] == "insert" and not (set(s_norm[raw_ops[position][3]:raw_ops[position][4]]) & deleted):
            raw_ops.pop(position)

    ops, quote_side, source_side, missing, extra, notes = [], [], [], [], [], []
    mismatches = 0
    orthographic = 0
    for position, (tag, i1, i2, j1, j2) in enumerate(raw_ops):
        source_idx = list(range(j1, j2))
        if tag == "replace" and _orthographic_variant(q_norm[i1:i2], s_norm[j1:j2], vocabulary):
            tag, orthographic = "equal", orthographic + 1   
        if tag == "replace" and position in (0, len(raw_ops) - 1) and len(raw_ops) > 1 and (j2 - j1) > (i2 - i1):
            keep = [j for j in source_idx if s_norm[j] in set(q_norm[i1:i2]) | deleted]
            source_idx = keep
            tag = "replace" if keep else "delete"
        ops.append({"op": tag, "span": " ".join(q_orig[i1:i2]), "source": " ".join(s_orig[j] for j in source_idx)})
        if tag == "equal":
            notes += _diacritic_notes(q_orig[i1:i2], s_orig[j1:j2], g_orig[j1:j2])
        else:
            mismatches += max(i2 - i1, len(source_idx))
            quote_side += q_norm[i1:i2]
            source_side += [s_norm[j] for j in source_idx]
            extra += q_orig[i1:i2] if tag in ("delete", "replace") else []
            missing += [s_orig[j] for j in source_idx] if tag in ("insert", "replace") else []

    matched = sum(i2 - i1 for tag, i1, i2, _, _ in raw_ops if tag == "equal")
    exact = bool(q_norm) and mismatches == 0
    return {
        "word_similarity": round(2 * matched / max(len(q_norm) + (raw_ops[-1][4] - raw_ops[0][3] if raw_ops else 0), 1), 3),
        "word_diff": ops,
        "missing_from_span": missing,
        "extra_in_span": extra,
        "source_excerpt": " ".join(op["source"] for op in ops if op["source"]),
        "exact": exact,
        "mismatches": mismatches,
        "allowed_gap": gap,
        "near": (not exact) and matched > 0 and mismatches <= gap,
        "reordered": bool(quote_side) and sorted(quote_side) == sorted(source_side),
        "diacritic_notes": notes,
        "orthographic_variants": orthographic,
    }
