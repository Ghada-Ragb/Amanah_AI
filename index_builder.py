from __future__ import annotations

import gzip
import json
import math
import pickle
import sys
import time
import zlib
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from normalization import content_words, normalize_for_matching, normalize_strict, phonetic_key, tokenize

INDEX_VERSION = 3
BM25_K1, BM25_B = 1.2, 0.75


class BM25Index:

    def __init__(self, postings: Dict[str, Tuple[array, array]], doc_len: array) -> None:
        self.postings, self.doc_len = postings, doc_len
        self.n_docs = len(doc_len)
        self.avg_len = (sum(doc_len) / self.n_docs) if self.n_docs else 1.0
        self.idf = {t: math.log(1 + (self.n_docs - len(d) + 0.5) / (len(d) + 0.5)) for t, (d, _) in postings.items()}

    def search(self, terms: Sequence[str], top_k: int) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = defaultdict(float)
        for term in set(terms):
            entry = self.postings.get(term)
            if entry is None:
                continue
            idf = self.idf[term]
            for doc, tf in zip(*entry):
                norm = 1 - BM25_B + BM25_B * self.doc_len[doc] / self.avg_len
                scores[doc] += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * norm)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


def _postings(documents: List[List[str]]) -> Tuple[Dict[str, Tuple[array, array]], array]:
    docs: Dict[str, array] = defaultdict(lambda: array("I"))
    tfs: Dict[str, array] = defaultdict(lambda: array("H"))
    doc_len = array("I")
    for doc_id, tokens in enumerate(documents):
        doc_len.append(len(tokens))
        for term, tf in Counter(tokens).items():
            docs[term].append(doc_id)
            tfs[term].append(min(tf, 65535))
    return {term: (docs[term], tfs[term]) for term in docs}, doc_len


def _read_json(path: Path) -> list:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected corpus format in {path}: expected a JSON list")
    return data



def anchor_keys(tokens: Sequence[str]) -> List[Tuple[int, int]]:
    keys = [phonetic_key(t) for t in tokens]
    out = []
    for i in range(len(keys) - 2):
        out.append((zlib.crc32(f"{keys[i]} {keys[i + 1]} {keys[i + 2]}".encode()), i))
        if i + 3 < len(keys):
            out.append((zlib.crc32(f"{keys[i]} {keys[i + 1]} _ {keys[i + 3]}".encode()), i))
            out.append((zlib.crc32(f"{keys[i]} _ {keys[i + 2]} {keys[i + 3]}".encode()), i))
    return out


def build_quran_index(path: Path) -> dict:
    records, norm, word_count = [], [], []
    all_index: Dict[str, List[int]] = defaultdict(list)
    by_surah: Dict[int, Dict[int, int]] = defaultdict(dict)
    documents: List[List[str]] = []
    anchors: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for entry in _read_json(Path(path)):
        text = (entry.get("ayah_text") or "").strip()
        if not text:
            continue
        idx = len(records)
        records.append({"surah_id": entry.get("surah_id"), "surah_name": entry.get("surah_name", ""),
                        "ayah_id": entry.get("ayah_id"), "text": text})
        by_surah[entry.get("surah_id")][entry.get("ayah_id")] = idx
        norm.append(normalize_for_matching(text))
        strict_tokens = tokenize(normalize_strict(text))
        word_count.append(len(strict_tokens))
        for word in set(strict_tokens):
            all_index[word].append(idx)
        documents.append(content_words(norm[-1].split()))
        for key, pos in anchor_keys(norm[-1].split()):
            anchors[key].append((idx, pos))
    postings, doc_len = _postings(documents)
    return {"version": INDEX_VERSION, "records": records, "norm": norm, "word_count": word_count,
            "all_index": dict(all_index), "by_surah": dict(by_surah), "postings": postings, "doc_len": doc_len,
            "anchors": dict(anchors)}


def build_hadith_index(path: Path) -> dict:
    records, documents = [], []
    for entry in _read_json(Path(path)):
        if not entry:
            continue
        matn = (entry.get("Matn") or "").strip() or None
        full = (entry.get("hadithTxt") or "").strip() or None
        if not (matn or full):
            continue

        records.append({"hadithID": entry.get("hadithID"), "book": entry.get("BookID"), "title": entry.get("title"),
                        "matn": matn, "full": None if matn else full})
        words = content_words(normalize_for_matching(full or matn).split())
        if matn and full:  
            extra = set(content_words(normalize_for_matching(matn).split())) - set(words)
            words += sorted(extra)
        documents.append(words)
    postings, doc_len = _postings(documents)
    return {"version": INDEX_VERSION, "records": records, "postings": postings, "doc_len": doc_len}


def save_index(index: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=9) as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        index = pickle.load(handle)   
    if index.get("version") != INDEX_VERSION:
        raise ValueError("Index version mismatch; rebuild with index_builder.py")
    return index


def main() -> None:
    base = Path(__file__).resolve().parent
    out = base / "index"
    for name, builder, source in (("quran", build_quran_index, base / "data" / "quran.json"),
                                  ("hadith", build_hadith_index, base / "data" / "hadith.json")):
        started = time.time()
        source = source if source.is_file() else source.with_name(source.name + ".gz")
        index = builder(source)
        save_index(index, out / f"{name}.idx.gz")
        size = (out / f"{name}.idx.gz").stat().st_size / 1e6
        print(f"{name}: {len(index['records'])} records -> index/{name}.idx.gz ({size:.1f} MB, {time.time() - started:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())
