from __future__ import annotations

import difflib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from alignment import align
from detector import DetectedSpan
from scanner import HybridDetector
from idgham import apply_idgham
from normalization import content_words, normalize_for_matching
from retrieval import SourceRetriever
from similarity import best_match_score, compute_signals

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 20_000
SHORT_QUOTE_WORDS = 3   
MIN_EXACT_TOKENS = 3   


@dataclass
class VerifierConfig:

    quran_correct_threshold: float = 0.94
    quran_uncertain_low: float = 0.45
    quran_min_coverage: float = 0.40
    hadith_correct_threshold: float = 0.88
    hadith_uncertain_low: float = 0.30
    hadith_min_coverage: float = 0.70
    quran_top_k: int = 25
    hadith_top_k: int = 15
    hadith_retrieval_guard: float = 0.20


@dataclass
class CorrectorConfig:

    max_window: int = 8
    hadith_top_k: int = 40
    quran_strong: float = 0.65
    min_full_ratio: float = 0.40
    quran_low: float = 0.55
    hadith_strong: float = 1.01
    hadith_low: float = 0.45


@dataclass
class PipelineConfig:
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    corrector: CorrectorConfig = field(default_factory=CorrectorConfig)
    verified_min_conf: float = 0.75        
    unsupported_min_conf: float = 0.70     
    unsupported_strength: float = 0.35     


@dataclass
class Verification:
    verdict: str                        
    confidence: float
    best_score: float
    method: str
    source: Optional[dict] = None        
    n_candidates: int = 0
    retrieval_top: float = 0.0


class Verifier:

    def __init__(self, retriever: SourceRetriever, config: Optional[VerifierConfig] = None) -> None:
        self.kb = retriever
        self.cfg = config or VerifierConfig()

    def verify(self, span_text: str, content_type: str) -> Verification:
        if not span_text or not span_text.strip():
            return self._result("Incorrect", 0.95, 0.0, None, 0, "empty_span")
        if content_type == "Ayah":
            return self._verify_quran(span_text)
        if content_type == "Hadith":
            return self._verify_hadith(span_text)
        return self._result("Incorrect", 0.5, 0.0, None, 0, "unknown_type")

    def _verify_quran(self, span: str) -> Verification:
        cfg = self.cfg
        candidates = self.kb.search_quran_ayahs(span, top_k=cfg.quran_top_k)
        if not candidates:
            return self._result("Incorrect", 0.8, 0.0, None, 0, "no_candidates")
        score, best = best_match_score(span, candidates, "Ayah")
        signals = best.get("signals", {}) if best else {}
        coverage, is_substring = signals.get("coverage", 0.0), signals.get("is_substring", 0)
        n = len(candidates)

        if is_substring and coverage >= cfg.quran_min_coverage:
            return self._result("Correct", min(0.98, 0.85 + score * 0.15), score, best, n, "substring_match")
        if score >= cfg.quran_correct_threshold and coverage >= cfg.quran_min_coverage:
            return self._result("Correct", min(0.95, 0.70 + score * 0.25), score, best, n, "threshold_pass")
        if score <= cfg.quran_uncertain_low:
            return self._result("Incorrect", min(0.95, 0.70 + (1 - score) * 0.25), score, best, n, "threshold_fail")

        strong = sum(
            1 for cand in candidates[:10]
            if (s := compute_signals(span, cand.get("text", ""), "Ayah"))["coverage"] >= 0.80 and s["lcs_ratio"] >= 0.75
        )
        if strong >= 2:
            return self._result("Correct", 0.60 + min(0.20, strong * 0.05), score, best, n, "borderline_multi_cov")
        return self._result("Incorrect", 0.58, score, best, n, "borderline_default")

    def _verify_hadith(self, span: str) -> Verification:
        cfg = self.cfg
        candidates = self.kb.search_hadith(span, top_k=cfg.hadith_top_k)
        if not candidates:
            return self._result("Incorrect", 0.75, 0.0, None, 0, "no_candidates")
        top_retrieval = candidates[0].get("retrieval_score", 0.0)
        score, best = best_match_score(span, candidates, "Hadith")
        signals = best.get("signals", {}) if best else {}
        coverage, is_substring = signals.get("coverage", 0.0), signals.get("is_substring", 0)
        n = len(candidates)

        if is_substring and coverage >= cfg.hadith_min_coverage and top_retrieval >= cfg.hadith_retrieval_guard:
            return self._result("Correct", min(0.97, 0.80 + score * 0.17), score, best, n, "substring_match", top_retrieval)
        if score >= cfg.hadith_correct_threshold and coverage >= cfg.hadith_min_coverage:
            return self._result("Correct", min(0.92, 0.65 + score * 0.27), score, best, n, "threshold_pass", top_retrieval)
        if score <= cfg.hadith_uncertain_low:
            return self._result("Incorrect", min(0.90, 0.65 + (1 - score) * 0.25), score, best, n, "threshold_fail", top_retrieval)

        moderate = sum(
            1 for cand in candidates[:8]
            if (s := compute_signals(span, cand.get("text", ""), "Hadith"))["coverage"] >= 0.65 and s["lcs_ratio"] >= 0.55
        )
        if moderate >= 2 and top_retrieval >= 0.30:
            return self._result("Correct", 0.58 + min(0.22, moderate * 0.06), score, best, n, "borderline_multi_cov", top_retrieval)
        if top_retrieval < 0.25 or score < 0.45:
            return self._result("Incorrect", 0.60, score, best, n, "borderline_low_retrieval", top_retrieval)
        return self._result("Incorrect", 0.55, score, best, n, "borderline_default", top_retrieval)

    @staticmethod
    def _result(verdict, confidence, score, best, n_candidates, method, top_retrieval=0.0) -> Verification:
        return Verification(verdict, round(confidence, 4), round(score, 4), method, best, n_candidates, round(top_retrieval, 4))


@dataclass
class CorrectionMatch:
    kind: str                  
    strength: float            
    full_ratio: float
    text: str                  
    source: dict               
    display: str = ""         

class Corrector:
    def __init__(self, retriever: SourceRetriever, config: Optional[CorrectorConfig] = None) -> None:
        self.kb = retriever
        self.cfg = config or CorrectorConfig()

    def match(self, span_text: str, content_type: str) -> Optional[CorrectionMatch]:
        return self.match_quran(span_text) if content_type == "Ayah" else self.match_hadith(span_text)

    def match_quran(self, query_text: str) -> Optional[CorrectionMatch]:
        """Best window of 1-8 consecutive ayahs (dynamic sliding window over BM25 seeds, exact-result pruning)."""
        kb = self.kb
        query_norm = normalize_for_matching(query_text)
        query_words = content_words(query_norm.split())
        if not query_words:
            return None
        query_len = len(query_norm)
        memo: Dict[tuple, tuple] = {}
        best = None   
        for seed in kb.quran_seed_ayahs(query_words, top_k=25):
            surah, ayah = kb.quran[seed]["surah_id"], kb.quran[seed]["ayah_id"]
            ayahs = kb.quran_by_surah[surah]
            min_ayah, max_ayah = min(ayahs), max(ayahs)
            for offset in range(3):
                start = ayah - offset
                if start < min_ayah:
                    continue
                window_len = -1
                for length in range(1, self.cfg.max_window + 1):
                    end = start + length - 1
                    if end > max_ayah:
                        break
                    window_len += len(kb.q_norm_match[ayahs[end]]) + 1
                    len_diff = abs(window_len - query_len)
                    upper_bound = min(1.0, window_len / max(query_len, 1))
                    if best is not None:   
                        best_cov, best_neg = best[0][0], best[0][1]
                        if upper_bound < best_cov or (upper_bound == best_cov and -len_diff < best_neg):
                            continue
                    key_pos = (surah, start, end)
                    if key_pos in memo:
                        continue
                    window = " ".join(kb.q_norm_match[ayahs[a]] for a in range(start, end + 1))
                    matcher = difflib.SequenceMatcher(None, query_norm, window, autojunk=False)
                    matched = sum(b.size for b in matcher.get_matching_blocks() if b.size >= 4)
                    coverage = matched / max(query_len, 1)
                    key = (coverage, -len_diff, matcher.ratio())
                    memo[key_pos] = key
                    if best is None or key > best[0]:
                        best = (key, coverage, key[2], surah, start, end)
        if best is None:
            return None
        _, coverage, ratio, surah, start, end = best
        display = " ".join(kb.quran[kb.quran_by_surah[surah][a]]["text"] for a in range(start, end + 1))
        return CorrectionMatch(
            "Ayah", coverage, ratio, self._ayah_text(surah, start, end),
            {"type": "Quran", "surah_id": surah, "surah_name": kb.quran[kb.quran_by_surah[surah][start]]["surah_name"],
             "ayah_start": start, "ayah_end": end},
            display,
        )

    def _ayah_text(self, surah: int, start: int, end: int) -> str:
        kb, multi = self.kb, end > start
        parts = []
        for a in range(start, end + 1):
            text = kb.quran[kb.quran_by_surah[surah][a]]["text"]
            parts.append(f"{text} ({a})" if multi else text)
        return apply_idgham(" ".join(parts)).replace("\u0640", "")

    def match_hadith(self, query_text: str) -> Optional[CorrectionMatch]:
        kb = self.kb
        query_norm = normalize_for_matching(query_text)
        query_words = content_words(query_norm.split())
        if not query_words:
            return None
        query_len = len(query_norm)
        best = None   
        for idx in kb.hadith_candidates(query_words, self.cfg.hadith_top_k):
            for field_name in ("matn", "full"):
                text = kb.hadith_norm(idx, field_name)
                if not text:
                    continue
                upper_bound = min(1.0, query_len / max(len(text), 1))
                if best is not None and upper_bound < best[0][0]:
                    continue
                matcher = difflib.SequenceMatcher(None, query_norm, text, autojunk=False)
                matched = sum(b.size for b in matcher.get_matching_blocks() if b.size >= 4)
                coverage, candidate_cov = matched / max(query_len, 1), matched / max(len(text), 1)
                key = (min(coverage, candidate_cov), matcher.ratio())
                if best is None or key > best[0]:
                    best = (key, idx, field_name, key[1])
        if best is None:
            return None
        key, idx, field_name, ratio = best
        record = kb.hadith[idx]
        text = record[field_name].strip()
        return CorrectionMatch(
            "Hadith", key[0], ratio, text,
            {"type": "Hadith", "hadithID": record["hadithID"], "book": record["book"], "title": record["title"],
             "field": "matn" if field_name == "matn" else "full_text"},
            text,
        )


STATUS_INFO = {
    "VERIFIED": {"ar": "موثّق", "group": "verified"},
    "CORRECTED": {"ar": "غير مطابق — يوجد تصحيح من المصدر", "group": "mismatch"},
    "UNSUPPORTED": {"ar": "غير مطابق — لا يوجد مصدر مطابق", "group": "mismatch"},
    "HUMAN_REVIEW": {"ar": "يحتاج مراجعة بشرية", "group": "review"},
}


class IslamicContentVerifier:

    def __init__(self, retriever: Optional[SourceRetriever] = None, config: Optional[PipelineConfig] = None,
                 use_scanner: bool = True, decouple_triggers: bool = True) -> None:
        self.cfg = config or PipelineConfig()
        self.retriever = retriever or SourceRetriever()
        self.verifier = Verifier(self.retriever, self.cfg.verifier)
        self.corrector = Corrector(self.retriever, self.cfg.corrector)
        self.detector = HybridDetector(self.retriever, use_scanner, rules_use_corpus=decouple_triggers,
                                       decouple_triggers=decouple_triggers)
        self.detector_name = type(self.detector).__name__


    def detect(self, text: str) -> List[DetectedSpan]:
        return self.detector.detect(self._validate(text)) if text.strip() else []

    def needs_hadith(self, text: str) -> bool:

        return any(span.label == "Hadith" for span in self.detect(text))

    def analyze(self, text: str) -> dict:

        text = self._validate(text)
        started = time.time()
        spans = self.detector.detect(text) if text.strip() else []
        detect_seconds = time.time() - started
        result = self._analyze_spans(text, spans)
        result["timings"] = {"detect_s": round(detect_seconds, 3), "total_s": round(time.time() - started, 3)}
        return result

    def analyze_spans(self, text: str, spans: List[dict]) -> dict:
        given = [DetectedSpan(s["start"], s["end"], s["label"], None, "given", text[s["start"]:s["end"]]) for s in spans]
        return self._analyze_spans(self._validate(text), given)

    def analyze_detected(self, text: str, spans: List[DetectedSpan]) -> dict:
        return self._analyze_spans(self._validate(text), spans)

    @staticmethod
    def _validate(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(f"Input is too long ({len(text)} characters); the limit is {MAX_INPUT_CHARS}")
        return text

    def _analyze_spans(self, text: str, spans: List[DetectedSpan]) -> dict:
        reports = [self._process_span(i + 1, span) for i, span in enumerate(sorted(spans, key=lambda s: s.start))]
        counts = {status: 0 for status in STATUS_INFO}
        for report in reports:
            counts[report["status"]] += 1
        return {
            "input_text": text,
            "detector": self.detector_name,
            "spans": reports,
            "corrected_text": self._apply_corrections(text, reports),
            "summary": {
                "n_spans": len(reports),
                "n_ayah": sum(r["type"] == "Ayah" for r in reports),
                "n_hadith": sum(r["type"] == "Hadith" for r in reports),
                **counts,
                "needs_human_review": counts["HUMAN_REVIEW"] > 0,
            },
        }

    def _process_span(self, index: int, span: DetectedSpan) -> dict:
        try:
            return self._final_verdict(self._ground(self._decide(index, span)))
        except Exception:   
            logger.exception("Failed to process span %d", index)
            report = self._empty_report(index, span)
            self._finalize(report, "HUMAN_REVIEW", {"code": "internal_error"})
            return report

    def _source_text(self, source: Optional[dict]) -> Optional[str]:
        kb = self.retriever
        if not source:
            return None
        if source.get("type") == "Quran":
            ayahs = kb.quran_by_surah.get(source.get("surah_id"), {})
            idxs = [ayahs.get(n) for n in range(source["ayah_start"], source["ayah_end"] + 1)]
            return " ".join(kb.q_norm_match[i] for i in idxs) if idxs and None not in idxs else None
        if source.get("type") == "Hadith":
            if getattr(self, "_hadith_by_id", None) is None:
                self._hadith_by_id = {}
                for record in kb.hadith:
                    self._hadith_by_id.setdefault((record["hadithID"], record["title"]), record)
            record = self._hadith_by_id.get((source.get("hadithID"), source.get("title")))
            if record is not None:
                return normalize_for_matching((record.get("matn") or "") + " " + (record.get("full") or ""))
        return None

    def _ground(self, report: dict) -> dict:

        evidence = report.get("evidence")
        if evidence and self._source_text(evidence["source"]) is None:
            report["evidence"], report["correction"], report["suggestion"] = None, None, None
            self._finalize(report, "HUMAN_REVIEW", {"code": "ungrounded"})
            report["grounding"] = "removed"
            return report
        for key in ("correction", "suggestion"):
            item = report.get(key)
            if not item:
                continue
            source_text = self._source_text(item["source"])
            shown = normalize_for_matching(item["display_text"])
            if source_text is None or shown not in source_text:
                report["correction"] = report["suggestion"] = None
                if report["status"] in ("CORRECTED", "VERIFIED"):
                    self._finalize(report, "HUMAN_REVIEW", {"code": "ungrounded"})
                report["grounding"] = "removed"
                return report
        report["grounding"] = "verified"
        return report

    @staticmethod
    def _final_verdict(report: dict) -> dict:
        report["verification"]["verdict"] = "Correct" if report["status"] == "VERIFIED" else "Incorrect"
        return report

    @staticmethod
    def _empty_report(index: int, span: DetectedSpan) -> dict:
        return {
            "id": index, "type": span.label, "start": span.start, "end": span.end, "text": span.text,
            "detection": {"backend": span.source, "confidence": None if span.confidence is None else round(span.confidence, 4)},
            "verification": {"verdict": "Incorrect", "confidence": 0.0, "score": 0.0, "method": "error", "n_candidates": 0},
            "evidence": None, "correction": None, "suggestion": None, "notes": [],
        }

    @staticmethod
    def _finalize(report: dict, status: str, reason: dict) -> None:
        info = STATUS_INFO[status]
        report.update(status=status, status_ar=info["ar"], group=info["group"], reason=reason)

    @staticmethod
    def source_label(source: dict) -> str:
        """Plain reference used in reasons, e.g. ``سورة البقرة 153`` or ``حديث رقم 5``."""
        if source["type"] == "Quran":
            start, end = source["ayah_start"], source["ayah_end"]
            return f"سورة {source['surah_name']} {start}" + (f"–{end}" if end != start else "")
        return f"حديث رقم {source['hadithID']}"

    def _decide(self, index: int, span: DetectedSpan) -> dict:
        report = self._empty_report(index, span)
        verification = self.verifier.verify(span.text, span.label)
        report["verification"] = {
            "verdict": verification.verdict, "confidence": verification.confidence, "score": verification.best_score,
            "method": verification.method, "n_candidates": verification.n_candidates,
        }
        whole = self._whole_ayah(span.text) if span.label == "Ayah" else None
        if whole is not None:
            self._verified_whole_ayah(report, span, whole)   
        elif span.label == "Ayah":
            self._decide_quran(report, span, verification)
        else:
            self._decide_hadith(report, span, verification)
        if whole is None and span.source == "rules" and len(normalize_for_matching(span.text).split()) < SHORT_QUOTE_WORDS:
            self._finalize(report, "HUMAN_REVIEW", {"code": "too_short"})
            report["correction"] = None
            report["suggestion"] = None
        elif report["status"] in ("UNSUPPORTED", "HUMAN_REVIEW"):
            self._cross_check(report, span)
        if report["status"] == "UNSUPPORTED":
            report["evidence"] = None   
        if report["status"] == "VERIFIED" and span.hint and span.hint != span.label:
            report["notes"].append({"code": "is_ayah" if span.label == "Ayah" else "is_hadith", "source": self.source_label(report["evidence"]["source"]), "misattributed": True})
        return report

    def _whole_ayah(self, text: str) -> Optional[int]:
        if getattr(self, "_whole_map", None) is None:
            mapping: Dict[str, int] = {}
            for i, norm in enumerate(self.retriever.q_norm_match):
                if len(norm.split()) >= 2:
                    mapping.setdefault(norm, i)
            self._whole_map = mapping
        norm = normalize_for_matching(text)
        return self._whole_map.get(norm) if len(norm.split()) >= 2 else None

    def _verified_whole_ayah(self, report: dict, span: DetectedSpan, idx: int) -> None:
        record = self.retriever.quran[idx]
        source_ref = {"type": "Quran", "surah_id": record["surah_id"], "surah_name": record["surah_name"],
                      "ayah_start": record["ayah_id"], "ayah_end": record["ayah_id"]}
        alignment = align(span.text, record["text"], self.retriever.quran_vocabulary)
        report["evidence"] = {"source": source_ref, "signals": compute_signals(span.text, record["text"], "Ayah"),
                              "comparison": alignment}
        report["verification"].update(verdict="Correct", confidence=0.98, score=1.0, method="whole_ayah")
        self._finalize(report, "VERIFIED", {"code": "exact_match", "source": self.source_label(source_ref)})
        if alignment["diacritic_notes"]:
            report["notes"].append({"code": "diacritic_conflict", "n": len(alignment["diacritic_notes"])})

    def _cross_check(self, report: dict, span: DetectedSpan) -> None:
        if len(normalize_for_matching(span.text).split()) < 3:
            return
        try:
            if span.label == "Ayah":
                other = self.verifier._verify_hadith(span.text)
                if other.verdict == "Correct" and other.method == "substring_match" and other.source:
                    report["notes"].append({"code": "is_hadith", "source": f"حديث رقم {other.source['hadithID']}"})
            else:
                other = self.verifier._verify_quran(span.text)
                if other.verdict == "Correct" and other.method == "substring_match" and other.source:
                    c = other.source
                    report["notes"].append({"code": "is_ayah", "source": f"سورة {c['surah_name']} {c['ayah_id']}"})
        except Exception:   
            logger.exception("cross-check failed")

    @staticmethod
    def _proposal(span: DetectedSpan, match: Optional[CorrectionMatch]) -> Optional[dict]:
        if match is None:
            return None
        return {"text": match.text, "display_text": match.display, "source": match.source,
                "match_strength": round(match.strength, 4), "full_ratio": round(match.full_ratio, 4)}

    def _decide_quran(self, report: dict, span: DetectedSpan, verification: Verification) -> None:
        cfg, corr_cfg = self.cfg, self.cfg.corrector
        match = self.corrector.match_quran(span.text)
        if match is not None:
            source_text, source_ref = match.display, match.source
        elif verification.source:
            candidate = verification.source
            source_text = candidate["text"]
            source_ref = {"type": "Quran", "surah_id": candidate["surah_id"], "surah_name": candidate["surah_name"],
                          "ayah_start": candidate["ayah_id"], "ayah_end": candidate["ayah_id"]}
        else:
            source_text, source_ref = "", None
        alignment = align(span.text, source_text, self.retriever.quran_vocabulary) if source_text else None
        if source_ref:
            report["evidence"] = {"source": source_ref, "signals": compute_signals(span.text, source_text, "Ayah"),
                                  "comparison": alignment}
        proposal = self._proposal(span, match)
        n_tokens = len(content_words(normalize_for_matching(span.text).split())) if span.text else 0
        has_tokens = alignment is not None and alignment["exact"] and len(alignment["word_diff"]) >= 1

        if has_tokens and sum(len(op["span"].split()) for op in alignment["word_diff"]) >= MIN_EXACT_TOKENS:
            self._finalize(report, "VERIFIED", {"code": "exact_match", "source": self.source_label(source_ref)})
            if alignment["diacritic_notes"]:
                report["notes"].append({"code": "diacritic_conflict", "n": len(alignment["diacritic_notes"])})
            if alignment["orthographic_variants"]:
                report["notes"].append({"code": "orthographic_variant", "n": alignment["orthographic_variants"]})
            return

        local = bool(alignment and alignment.get("near") and alignment.get("word_similarity", 0) >= 0.75 and alignment.get("source_excerpt"))
        strong = match is not None and match.strength >= corr_cfg.quran_strong and (match.full_ratio >= corr_cfg.min_full_ratio or local)
        if strong:
            self._finalize(report, "CORRECTED", {"code": "altered_passage", "source": self.source_label(source_ref),
                                                 "n": alignment["mismatches"] if alignment else 0,
                                                 "reordered": bool(alignment and alignment["reordered"])})
            excerpt = alignment["source_excerpt"] if alignment and alignment["source_excerpt"] else proposal["display_text"]
            report["correction"] = {**proposal, "display_text": excerpt, "applied": True}
            return

        if verification.verdict == "Correct":
            self._finalize(report, "HUMAN_REVIEW", {"code": "weak_match"})
            report["suggestion"] = proposal
        elif match is None or match.strength < corr_cfg.quran_low:
            confident = verification.confidence >= cfg.unsupported_min_conf and not verification.method.startswith("borderline")
            if match is None or match.strength < cfg.unsupported_strength or confident:
                self._finalize(report, "UNSUPPORTED", {"code": "no_source"})
            else:
                self._finalize(report, "HUMAN_REVIEW", {"code": "insufficient_evidence"})
                report["suggestion"] = proposal
        else:
            self._finalize(report, "HUMAN_REVIEW", {"code": "candidate_not_strong", "source": self.source_label(source_ref),
                                                    "strength": round(match.strength, 2)})
            report["suggestion"] = proposal

    def _decide_hadith(self, report: dict, span: DetectedSpan, verification: Verification) -> None:
        cfg, corr_cfg = self.cfg, self.cfg.corrector
        best = verification.source
        alignment = None
        if best:
            alignment = align(span.text, best["text"])
            report["evidence"] = {
                "source": {"type": "Hadith", "hadithID": best["hadithID"], "book": best["book"], "title": best["title"]},
                "signals": best.get("signals"), "comparison": alignment,
            }
        exact_tokens = alignment is not None and alignment["exact"] and sum(len(op["span"].split()) for op in alignment["word_diff"]) >= 4

        if verification.verdict == "Correct" or exact_tokens:
            altered = (not exact_tokens and alignment is not None and not alignment["exact"] and alignment["mismatches"] >= 1)
            if altered:   
                match = self.corrector.match_hadith(span.text)
                self._finalize(report, "HUMAN_REVIEW", {"code": "hadith_altered", "n": alignment["mismatches"],
                                                        "source": f"حديث رقم {best['hadithID']}"})
                report["suggestion"] = self._proposal(span, match)
                report["verification"]["verdict"] = "Incorrect"
            elif exact_tokens or (not verification.method.startswith("borderline") and verification.confidence >= cfg.verified_min_conf):
                self._finalize(report, "VERIFIED", {"code": "exact_match" if (exact_tokens or (alignment and alignment["exact"])) else "close_match",
                                                    "source": f"حديث رقم {best['hadithID']}"})
                if alignment and not alignment["exact"]:
                    report["notes"].append({"code": "hadith_minor_diffs", "n": alignment["mismatches"]})
                if alignment and alignment["diacritic_notes"]:
                    report["notes"].append({"code": "diacritic_conflict", "n": len(alignment["diacritic_notes"])})
            else:
                self._finalize(report, "HUMAN_REVIEW", {"code": "weak_match"})
            return

        match = self.corrector.match_hadith(span.text)
        proposal = self._proposal(span, match)
        if match is None or match.strength < corr_cfg.hadith_low:
            confident = verification.confidence >= cfg.unsupported_min_conf and not verification.method.startswith("borderline")
            if match is None or match.strength < cfg.unsupported_strength or confident:
                self._finalize(report, "UNSUPPORTED", {"code": "no_source"})
            else:
                self._finalize(report, "HUMAN_REVIEW", {"code": "insufficient_evidence"})
                report["suggestion"] = proposal
        else:   
            self._finalize(report, "HUMAN_REVIEW", {"code": "hadith_candidate", "source": self.source_label(match.source),
                                                    "strength": round(match.strength, 2)})
            report["suggestion"] = proposal

    @staticmethod
    def _apply_corrections(text: str, reports: List[dict]) -> str:
        out = text
        for report in sorted(reports, key=lambda r: r["start"], reverse=True):
            if report["status"] == "CORRECTED" and report["correction"] and report["correction"].get("applied"):
                out = out[: report["start"]] + report["correction"]["display_text"] + out[report["end"]:]
        return out
