from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from index_builder import BM25Index, build_hadith_index, build_quran_index, load_index, save_index
from normalization import char_ngrams, content_words, normalize_for_matching, normalize_lenient, normalize_strict, tokenize

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"


class CorpusError(RuntimeError):
    """Raised when a corpus or index file is missing or malformed."""


class SourceRetriever:

    def __init__(self, index_dir: Path = INDEX_DIR, data_dir: Path = DATA_DIR) -> None:
        self.index_dir, self.data_dir = Path(index_dir), Path(data_dir)
        self._hadith_lock = threading.Lock()
        self._hadith: Optional[dict] = None
        self._norm_cache: "OrderedDict[Tuple[int, str], str]" = OrderedDict()

        quran = self._load("quran", build_quran_index)
        self.quran: List[dict] = quran["records"]
        self.q_norm_match: List[str] = quran["norm"]
        self.q_word_count: List[int] = quran["word_count"]
        self.q_all_index: Dict[str, List[int]] = quran["all_index"]
        self.quran_by_surah: Dict[int, Dict[int, int]] = quran["by_surah"]
        self.quran_bm25 = BM25Index(quran["postings"], quran["doc_len"])
        self.quran_anchors: Dict[int, List[Tuple[int, int]]] = quran["anchors"]
        logger.info("Quran index ready: %d ayahs", len(self.quran))

    def _load(self, name: str, builder) -> dict:
        path = self.index_dir / f"{name}.idx.gz"
        if path.is_file():
            try:
                return load_index(path)
            except Exception:
                logger.warning("Index %s is unreadable; rebuilding from data/", path)
        source = self.data_dir / ("quran.json" if name == "quran" else "hadith.json")
        gz = source.with_name(source.name + ".gz")
        source = source if source.is_file() else gz
        if not source.is_file():
            raise CorpusError(f"Neither {path} nor the source corpus {source} was found")
        index = builder(source)
        try:
            save_index(index, path)
        except OSError:
            logger.info("Could not cache %s (read-only file system); continuing in memory", path)
        return index

    def warm(self) -> None:
        _ = self.hadith_data

    @property
    def hadith_loaded(self) -> bool:
        return self._hadith is not None

    @property
    def hadith_data(self) -> dict:
        if self._hadith is None:
            with self._hadith_lock:
                if self._hadith is None:
                    data = self._load("hadith", build_hadith_index)
                    data["bm25"] = BM25Index(data["postings"], data["doc_len"])
                    self._hadith = data
                    logger.info("Hadith index ready: %d records", len(data["records"]))
        return self._hadith

    @property
    def hadith(self) -> List[dict]:
        return self.hadith_data["records"]

    @property
    def quran_vocabulary(self) -> frozenset:
        if getattr(self, "_vocab", None) is None:
            self._vocab = frozenset(word for text in self.q_norm_match for word in text.split())
        return self._vocab

    def search_quran_ayahs(self, query: str, top_k: int = 25, extra_bm25: int = 10) -> List[dict]:
        query_words = tokenize(normalize_strict(query))
        if not query_words:
            return []
        votes: Dict[int, int] = {}
        for word in query_words:
            for idx in self.q_all_index.get(word, ()):
                votes[idx] = votes.get(idx, 0) + 1
        scored: List[Tuple[int, float]] = []
        for idx, vote in votes.items():
            coverage = vote / len(query_words)
            precision = vote / self.q_word_count[idx] if self.q_word_count[idx] else 0.0
            f1 = 2 * coverage * precision / (coverage + precision) if coverage + precision > 0 else 0.0
            scored.append((idx, f1))
        scored.sort(key=lambda item: item[1], reverse=True)
        scores = dict(scored)
        ranked = [idx for idx, _ in scored[:top_k]]
        seen = set(ranked)
        bm25_hits = self.quran_bm25.search(content_words(normalize_for_matching(query).split()), extra_bm25)
        ranked += [idx for idx, _ in bm25_hits if idx not in seen]
        results = []
        for idx in ranked:
            candidate = dict(self.quran[idx])
            candidate.update(type="Quran", retrieval_score=scores.get(idx, 0.0))
            results.append(candidate)
        return results

    def quran_seed_ayahs(self, query_words: Sequence[str], top_k: int = 25) -> List[int]:
        return [idx for idx, _ in self.quran_bm25.search(query_words, top_k)]

    def hadith_candidates(self, query_words: Sequence[str], top_k: int) -> List[int]:
        return [idx for idx, _ in self.hadith_data["bm25"].search(query_words, top_k)]

    def hadith_norm(self, idx: int, field: str) -> Optional[str]:
        record = self.hadith[idx]
        raw = record.get(field)
        if not raw:
            return None
        key = (idx, field)
        if key in self._norm_cache:
            self._norm_cache.move_to_end(key)
            return self._norm_cache[key]
        value = normalize_for_matching(raw)
        self._norm_cache[key] = value
        if len(self._norm_cache) > 4096:
            self._norm_cache.popitem(last=False)
        return value

    def search_hadith(self, query: str, top_k: int = 15, pool: int = 60) -> List[dict]:
        words = content_words(normalize_for_matching(query).split())
        if not words:
            return []
        query_grams = char_ngrams(normalize_lenient(query))
        results = []
        for idx in self.hadith_candidates(words, pool):
            record = self.hadith[idx]
            text = record["matn"] or record["full"]
            doc_grams = char_ngrams(normalize_lenient(text))
            cosine = (
                len(query_grams & doc_grams) / ((len(query_grams) * len(doc_grams)) ** 0.5)
                if query_grams and doc_grams
                else 0.0
            )
            results.append(
                {
                    "type": "Hadith",
                    "idx": idx,
                    "hadithID": record["hadithID"],
                    "book": record["book"],
                    "title": record["title"],
                    "text": text,
                    "has_matn": bool(record["matn"]),
                    "retrieval_score": cosine,
                }
            )
        results.sort(key=lambda c: c["retrieval_score"], reverse=True)
        return results[:top_k]
