from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional

HF_ENDPOINT = "https://router.huggingface.co/hf-inference/models/{model}"   

LABEL_ALIASES = {"ayah": "Ayah", "quran": "Ayah", "label_1": "Ayah", "label_2": "Ayah",
                 "hadith": "Hadith", "label_3": "Hadith", "label_4": "Hadith"}
DEFAULT_MIN_SCORE = 0.5


def _label_of(raw: str) -> Optional[str]:
    name = re.sub(r"^[BI]-", "", str(raw or ""), flags=re.I).strip().lower()
    return LABEL_ALIASES.get(name)


def entities_to_spans(text: str, entities: Iterable[dict], min_score: float = DEFAULT_MIN_SCORE,
                      max_gap: int = 1) -> List[dict]:
    pieces = []
    for item in entities or []:
        label = _label_of(item.get("entity_group") or item.get("entity"))
        start, end = item.get("start"), item.get("end")
        if label is None or start is None or end is None or end <= start:
            continue
        pieces.append({"label": label, "start": int(start), "end": int(end), "score": float(item.get("score", 1.0)),
                       "begin": str(item.get("entity", "")).upper().startswith("B-")})
    pieces.sort(key=lambda p: (p["start"], p["end"]))
    merged: List[dict] = []
    for piece in pieces:
        last = merged[-1] if merged else None
        if last and last["label"] == piece["label"] and not piece["begin"] and piece["start"] - last["end"] <= max_gap:
            last["end"] = max(last["end"], piece["end"])
            last["scores"].append(piece["score"])
        else:
            merged.append({"label": piece["label"], "start": piece["start"], "end": piece["end"], "scores": [piece["score"]]})
    spans = []
    for item in merged:
        start, end = item["start"], item["end"]
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        score = sum(item["scores"]) / len(item["scores"])
        if end > start and score >= min_score:
            spans.append({"label": item["label"], "start": start, "end": end, "score": round(score, 4)})
    return spans


def merge_spans(text: str, detected: list, model_spans: List[dict], min_words: int = 3) -> list:
    from detector import DetectedSpan

    merged = list(detected)
    for item in sorted(model_spans or [], key=lambda s: s["start"]):
        start, end = int(item["start"]), int(item["end"])
        if end <= start or end > len(text) or len(text[start:end].split()) < min_words:
            continue
        if any(start < d.end and end > d.start for d in merged):
            continue
        merged.append(DetectedSpan(start, end, item["label"], item.get("score"), "camelbert", text[start:end]))
    return sorted(merged, key=lambda s: s.start)


def analyze_hybrid(pipeline, text: str, model_spans: Optional[List[dict]] = None) -> dict:
    detected = pipeline.detect(text)
    spans = merge_spans(text, detected, model_spans or [])
    result = pipeline.analyze_detected(text, spans)
    result["detector"] = "hybrid" if model_spans else pipeline.detector_name
    return result


def query_hosted_model(text: str, model: str, token: str = "", endpoint: str = HF_ENDPOINT, timeout: float = 30.0) -> List[dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps({"inputs": text, "parameters": {"aggregation_strategy": "simple"}}).encode("utf-8")
    request = urllib.request.Request(endpoint.format(model=model), data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("hosted model unavailable") from exc
    if not isinstance(payload, list):
        raise RuntimeError("unexpected response from the hosted model")
    return entities_to_spans(text, payload)
