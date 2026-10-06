from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from alignment import aligned_words
from detector import DetectedSpan, RuleDetector, trim_span
from index_builder import anchor_keys
from normalization import content_words, normalize_for_matching, phonetic_key
from retrieval import SourceRetriever

_ARABIC = re.compile(r"[\u0621-\u064A]")
_STRONG_BOUNDARY = re.compile(r"[.؟?!؛;:\n…]")


@dataclass
class Token:
    skeleton: str
    key: str
    start: int
    end: int
    boundary_after: bool   


def tokenize_with_offsets(text: str) -> List[Token]:
    matches = list(re.finditer(r"\S+", text))
    tokens: List[Token] = []
    for i, match in enumerate(matches):
        skeleton = normalize_for_matching(match.group()).replace(" ", "")
        if not skeleton or not _ARABIC.search(skeleton):
            if tokens and _STRONG_BOUNDARY.search(match.group()):
                tokens[-1].boundary_after = True
            continue
        gap_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        trailing = text[match.end():gap_end] + match.group()[-2:]
        tokens.append(Token(skeleton, phonetic_key(skeleton), match.start(), match.end(),
                            bool(_STRONG_BOUNDARY.search(trailing))))
    return tokens


@dataclass
class _Region:
    start: int          
    end: int            
    label: str
    matched: int
    ratio: float
    idf: float
    surah: Optional[int] = None
    first_ayah: Optional[int] = None
    last_ayah: Optional[int] = None


class CorpusScanner:

    def __init__(self, retriever: SourceRetriever, min_tokens: int = 4, min_ratio: float = 0.7, min_idf: float = 8.0,
                 hadith_min_tokens: int = 6, hadith_min_idf: float = 14.0, scan_hadith: bool = True) -> None:
        self.kb = retriever
        self.min_tokens, self.min_ratio, self.min_idf = min_tokens, min_ratio, min_idf
        self.hadith_min_tokens, self.hadith_min_idf = hadith_min_tokens, hadith_min_idf
        self.scan_hadith = scan_hadith

    def scan(self, text: str, exclude: Sequence[Tuple[int, int]] = ()) -> List[DetectedSpan]:
        tokens = tokenize_with_offsets(text)
        runs = self._runs(tokens, exclude)
        spans: List[DetectedSpan] = []
        for run in runs:
            for region in self._merge(self._scan_quran(run)):
                spans.append(self._to_span(text, run, region))
        if self.scan_hadith:
            for run in runs:
                taken = [(s.start, s.end) for s in spans]
                for region in self._scan_hadith(run, taken):
                    spans.append(self._to_span(text, run, region))
        return sorted(spans, key=lambda s: s.start)

    @staticmethod
    def _runs(tokens: List[Token], exclude: Sequence[Tuple[int, int]]) -> List[List[Token]]:
        runs, current = [], []
        for token in tokens:
            if any(token.start < e and token.end > s for s, e in exclude):
                if current:
                    runs.append(current)
                current = []
            else:
                current.append(token)
        if current:
            runs.append(current)
        return runs

    @staticmethod
    def _to_span(text: str, run: List[Token], region: _Region) -> DetectedSpan:
        start, end = trim_span(text, run[region.start].start, run[region.end - 1].end)
        return DetectedSpan(start, end, region.label, round(region.ratio, 3), "scan", text[start:end])

    def _scan_quran(self, run: List[Token]) -> List[_Region]:
        kb = self.kb
        if len(run) < self.min_tokens:
            return []
        hits: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for key_hash, pos in anchor_keys([t.skeleton for t in run]):
            postings = kb.quran_anchors.get(key_hash)
            if not postings or len(postings) > 60:   
                continue
            for ayah, source_pos in postings:
                hits[ayah].append((pos, source_pos))

        regions: List[_Region] = self._whole_ayahs(run)
        for ayah, found in hits.items():
            found.sort()
            chains: List[dict] = []
            for pos, source_pos in found:
                diagonal = source_pos - pos
                for chain in chains:
                    if abs(diagonal - chain["d"]) <= 2 and pos - chain["last"] <= 5:
                        chain["hits"].append((pos, source_pos))
                        chain["last"], chain["d"] = pos, diagonal
                        break
                else:
                    chains.append({"d": diagonal, "last": pos, "hits": [(pos, source_pos)]})
            for chain in chains:
                region = self._grow(run, ayah, chain)
                if region is not None:
                    regions.append(region)
        return self._non_overlapping(regions)

    def _whole_ayah_index(self) -> Dict[str, list]:
        if getattr(self, "_whole", None) is None:
            index: Dict[str, list] = defaultdict(list)
            for i, norm in enumerate(self.kb.q_norm_match):
                words = tuple(w for w in norm.split() if w)
                if len(words) >= self.min_tokens:
                    index[words[0]].append((words, i))
            for entries in index.values():
                entries.sort(key=lambda e: -len(e[0]))   
            self._whole = index
        return self._whole

    def _whole_ayahs(self, run: List[Token]) -> List[_Region]:
        index, skeletons, found, i = self._whole_ayah_index(), [t.skeleton for t in run], [], 0
        while i < len(run):
            for words, ayah in index.get(skeletons[i], ()):
                n = len(words)
                if tuple(skeletons[i:i + n]) == words and not any(t.boundary_after for t in run[i:i + n - 1]):
                    record = self.kb.quran[ayah]
                    found.append(_Region(i, i + n, "Ayah", n, 1.0, 99.0, record["surah_id"], record["ayah_id"], record["ayah_id"]))
                    i += n - 1
                    break
            i += 1
        return found

    def _grow(self, run: List[Token], ayah: int, chain: dict) -> Optional[_Region]:
        kb = self.kb
        source = kb.q_norm_match[ayah].split()
        source_keys = [phonetic_key(w) for w in source]
        diagonal = sorted(source_pos - pos for pos, source_pos in chain["hits"])[len(chain["hits"]) // 2]
        start = min(pos for pos, _ in chain["hits"])
        end = min(len(run), max(pos for pos, _ in chain["hits"]) + 3)

        def key_at(i: int) -> Optional[str]:
            return source_keys[i + diagonal] if 0 <= i + diagonal < len(source_keys) else None

        while start > 0 and not run[start - 1].boundary_after:   
            if run[start - 1].key == key_at(start - 1):
                start -= 1
            elif start >= 2 and not run[start - 2].boundary_after and run[start - 2].key == key_at(start - 2):
                start -= 2   
            else:
                break
        while end < len(run) and not run[end - 1].boundary_after:
            if run[end].key == key_at(end):
                end += 1
            elif end + 1 < len(run) and run[end + 1].key == key_at(end + 1):
                end += 2
            else:
                break

        start = self._soft_left(run, start, diagonal, source_keys)
        end = self._soft_right(run, end, diagonal, source_keys)
        positions = range(start, end)
        matched_idx = [i for i in positions if run[i].key == key_at(i)]
        matched = len(matched_idx)
        n = end - start
        idf = sum(kb.quran_bm25.idf.get(run[i].skeleton, 0.0) for i in matched_idx)
        ratio = matched / n if n else 0.0
        needed_idf = self.min_idf if n >= 5 else self.min_idf + 6.0   
        if n < self.min_tokens or matched < self.min_tokens or ratio < self.min_ratio or idf < needed_idf:
            return None
        record = kb.quran[ayah]
        return _Region(start, end, "Ayah", matched, ratio, idf, record["surah_id"], record["ayah_id"], record["ayah_id"])

    @staticmethod
    def _closest(run: List[Token], candidates, target: str):
        scored = [(SequenceMatcher(None, "".join(t.key for t in run[a:b]), target).ratio(), a, b) for a, b in candidates]
        if not scored:
            return None
        top = max(score for score, _, _ in scored)
        if top < 0.6:
            return None
        return max((c for c in scored if c[0] >= top - 0.2), key=lambda c: c[2] - c[1])

    def _soft_left(self, run: List[Token], start: int, diagonal: int, source_keys: List[str]) -> int:
        missing = min(start + diagonal, 2)
        if missing <= 0:
            return start
        target = "".join(source_keys[start + diagonal - missing : start + diagonal])
        candidates = [(start - n, start) for n in range(max(1, missing - 1), missing + 2)
                      if start - n >= 0 and not any(run[i].boundary_after for i in range(start - n, start))]
        best = self._closest(run, candidates, target)
        return best[1] if best else start

    def _soft_right(self, run: List[Token], end: int, diagonal: int, source_keys: List[str]) -> int:
        missing = min(len(source_keys) - (end + diagonal), 2)
        if missing <= 0 or end >= len(run) or run[end - 1].boundary_after:
            return end
        target = "".join(source_keys[end + diagonal : end + diagonal + missing])
        candidates = [(end, end + n) for n in range(max(1, missing - 1), missing + 2)
                      if end + n <= len(run) and not any(run[i].boundary_after for i in range(end, end + n - 1))]
        best = self._closest(run, candidates, target)
        return best[2] if best else end

    @staticmethod
    def _non_overlapping(regions: List[_Region]) -> List[_Region]:
        chosen: List[_Region] = []
        for region in sorted(regions, key=lambda r: (r.matched, r.ratio), reverse=True):
            if all(region.end <= c.start or region.start >= c.end for c in chosen):
                chosen.append(region)
        return sorted(chosen, key=lambda r: r.start)

    @staticmethod
    def _merge(regions: List[_Region]) -> List[_Region]:
        merged: List[_Region] = []
        for region in regions:
            last = merged[-1] if merged else None
            if (last and last.surah == region.surah and region.first_ayah == last.last_ayah + 1
                    and region.start - last.end <= 1):
                last.end, last.matched = region.end, last.matched + region.matched
                last.ratio = last.matched / (last.end - last.start)
                last.idf += region.idf
                last.last_ayah = region.last_ayah
            else:
                merged.append(region)
        return merged

    def _scan_hadith(self, run: List[Token], taken: Sequence[Tuple[int, int]]) -> List[_Region]:
        kb = self.kb
        regions: List[_Region] = []
        segment_start = 0
        for i, token in enumerate(run):
            if token.boundary_after or i == len(run) - 1:
                segment = (segment_start, i + 1)
                segment_start = i + 1
                if segment[1] - segment[0] < self.hadith_min_tokens:
                    continue
                for a, b in self._windows(*segment):
                    region = self._match_hadith(run, a, b)
                    if region is not None:
                        regions.append(region)
        return self._non_overlapping(regions)

    @staticmethod
    def _windows(start: int, end: int, size: int = 24, stride: int = 12):
        if end - start <= 40:
            yield start, end
        else:
            for a in range(start, end - 6, stride):
                yield a, min(end, a + size)

    def _match_hadith(self, run: List[Token], a: int, b: int) -> Optional[_Region]:
        kb = self.kb
        window = run[a:b]
        words = content_words(t.skeleton for t in window)
        if len(words) < 4:
            return None
        window_keys = [t.key for t in window]
        best = None
        for idx in kb.hadith_candidates(words, 3):
            record = kb.hadith[idx]
            source = [p[1] for p in aligned_words(record["matn"] or record["full"])]
            matcher = SequenceMatcher(None, window_keys, [phonetic_key(w) for w in source], autojunk=False)
            blocks = [blk for blk in matcher.get_matching_blocks() if blk.size >= 3]
            matched = sum(blk.size for blk in blocks)
            if matched >= self.hadith_min_tokens and (best is None or matched > best[0]):
                best = (matched, blocks)
        if best is None:
            return None
        matched, blocks = best
        start, end = blocks[0].a, blocks[-1].a + blocks[-1].size
        ratio = matched / (end - start)
        idf = sum(kb.hadith_data["bm25"].idf.get(window[i].skeleton, 0.0) for blk in blocks for i in range(blk.a, blk.a + blk.size))
        if ratio < self.min_ratio or idf < self.hadith_min_idf:
            return None
        return _Region(a + start, a + end, "Hadith", matched, ratio, idf)


class HybridDetector:

    def __init__(self, retriever: SourceRetriever, use_scanner: bool = True, rules_use_corpus: bool = True,
                 decouple_triggers: bool = True) -> None:
        self.rules = RuleDetector(retriever if rules_use_corpus else None, decouple_triggers=decouple_triggers)
        self.scanner = CorpusScanner(retriever) if use_scanner else None

    def detect(self, text: str) -> List[DetectedSpan]:
        spans = self.rules.detect(text)
        if self.scanner is not None:
            found = sorted(self.scanner.scan(text, [(s.start, s.end) for s in spans]), key=lambda s: s.end - s.start, reverse=True)
            kept: List[DetectedSpan] = []
            for span in found:   
                if all(span.end <= k.start or span.start >= k.end for k in kept):
                    kept.append(span)
            spans += kept
        return sorted(spans, key=lambda s: s.start)
