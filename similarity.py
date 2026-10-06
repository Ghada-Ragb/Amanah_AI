from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

from alignment import best_region, lcs_length
from normalization import normalize_lenient, normalize_strict, tokenize

try:  
    from rapidfuzz.distance import Levenshtein as _RapidLevenshtein
except ImportError:  
    _RapidLevenshtein = None


def _edit_similarity(a: str, b: str, max_len: int = 600) -> float:
    a, b = a[:max_len], b[:max_len]
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if _RapidLevenshtein is not None:
        return float(_RapidLevenshtein.normalized_similarity(a, b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (ca != cb)))
        prev = curr
    return 1.0 - prev[-1] / max(len(a), len(b))


def _light_normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[،؛؟!.,:;'\"()\[\]{}<>«»\-_/\\|@#$%^&*+=~`\u0640]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_signals(claim: str, candidate: str, content_type: str = "Ayah", local: bool = True) -> Dict[str, float]:
    is_quran = content_type == "Ayah"
    full_candidate = candidate
    if local:   
        candidate = best_region(claim, candidate)
    normalize = normalize_strict if is_quran else normalize_lenient
    norm_claim, norm_cand = normalize(claim), normalize(candidate)

    claim_set, cand_set = set(tokenize(norm_claim)), set(tokenize(norm_cand))
    if claim_set and cand_set:
        shared = claim_set & cand_set
        token_overlap = len(shared) / len(claim_set | cand_set)
        coverage = len(shared) / len(claim_set)
    else:
        token_overlap = coverage = 0.0

    claim_tokens, cand_tokens = tokenize(norm_claim), tokenize(norm_cand)
    lcs_ratio = lcs_length(claim_tokens, cand_tokens) / len(claim_tokens) if claim_tokens else 0.0
    edit_sim = _edit_similarity(norm_claim, norm_cand)

    claim_chars, cand_chars = set(norm_claim.replace(" ", "")), set(norm_cand.replace(" ", ""))
    char_overlap = len(claim_chars & cand_chars) / len(claim_chars | cand_chars) if (claim_chars or cand_chars) else 0.0

    claim_flat, cand_flat = norm_claim.replace(" ", ""), norm_cand.replace(" ", "")
    is_substring = int(bool(claim_flat) and bool(cand_flat) and (claim_flat in cand_flat or cand_flat in claim_flat))

    diacritic_sim = (
        _edit_similarity(_light_normalize(claim), _light_normalize(candidate), max_len=800) if is_quran else edit_sim
    )
    short = len(claim_tokens) < 4

    if is_quran:
        w = (
            dict(coverage=0.35, diacritic_sim=0.30, lcs_ratio=0.15, token_overlap=0.10, char_overlap=0.05, edit_sim=0.05)
            if short
            else dict(coverage=0.25, diacritic_sim=0.30, lcs_ratio=0.20, token_overlap=0.10, char_overlap=0.05, edit_sim=0.10)
        )
        composite = (
            w["coverage"] * coverage + w["diacritic_sim"] * diacritic_sim + w["lcs_ratio"] * lcs_ratio
            + w["token_overlap"] * token_overlap + w["char_overlap"] * char_overlap + w["edit_sim"] * edit_sim
        )
        if is_substring and diacritic_sim >= 0.60:
            composite = max(composite, 0.88)
        elif is_substring:
            composite = max(composite, 0.75)
    else:
        w = (
            dict(coverage=0.40, lcs_ratio=0.20, token_overlap=0.20, char_overlap=0.10, edit_sim=0.10)
            if short
            else dict(coverage=0.30, lcs_ratio=0.28, token_overlap=0.18, char_overlap=0.12, edit_sim=0.12)
        )
        composite = (
            w["coverage"] * coverage + w["lcs_ratio"] * lcs_ratio + w["token_overlap"] * token_overlap
            + w["char_overlap"] * char_overlap + w["edit_sim"] * edit_sim
        )
        if is_substring:
            composite = max(composite, 0.82)

    return {
        "token_overlap": round(token_overlap, 4),
        "coverage": round(coverage, 4),
        "lcs_ratio": round(lcs_ratio, 4),
        "edit_sim": round(edit_sim, 4),
        "diacritic_sim": round(diacritic_sim, 4),
        "char_overlap": round(char_overlap, 4),
        "is_substring": is_substring,
        "source_fraction": round(min(1.0, len(claim_tokens) / max(len(tokenize(normalize(full_candidate))), 1)), 4),
        "composite": round(composite, 4),
    }


def best_match_score(claim: str, candidates: List[dict], content_type: str = "Ayah", local: bool = True):
    """Return ``(score, candidate_with_signals)`` for the best-scoring candidate."""
    best_score, best_candidate = 0.0, None
    for candidate in candidates:
        signals = compute_signals(claim, candidate.get("text", ""), content_type, local)
        if signals["composite"] > best_score:
            best_score, best_candidate = signals["composite"], {**candidate, "signals": signals}
    return best_score, best_candidate
