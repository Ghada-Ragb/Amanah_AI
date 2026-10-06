from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

NO_SOURCE = "خطأ"
FILE_NAMES = ("task1A_predictions.tsv", "task1B_predictions.tsv", "task1C_predictions.tsv")
Row = Tuple


def _clean(value) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def competition_rows(result: dict, qid: str) -> Tuple[List[Row], List[Row], List[Row]]:
    a, b, c = [], [], []
    spans = result["spans"]
    if not spans:
        a.append((qid, 0, 0, "No_Spans"))
    for k, span in enumerate(spans, 1):
        a.append((qid, span["start"], span["end"], span["type"]))
        verdict = span["verification"]["verdict"]
        b.append((f"{qid}_{k}", verdict))
        if verdict == "Incorrect":
            correction = span["correction"]["text"] if (span["status"] == "CORRECTED" and span["correction"]) else NO_SOURCE
            c.append((f"{qid}_{span['start']}_{span['end']}", _clean(correction)))
    return a, b, c


def to_tsv(rows: Iterable[Row]) -> str:
    return "".join("\t".join(str(x) for x in row) + "\n" for row in rows)


def official_payload(result: dict, qid: str = "Q1") -> Dict[str, str]:
    return {name: to_tsv(rows) for name, rows in zip(FILE_NAMES, competition_rows(result, qid))}


def write_competition_files(all_rows: Sequence[Sequence[Row]], out_dir: str = "outputs") -> List[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in zip(FILE_NAMES, all_rows):
        path = out / name
        path.write_text(to_tsv(rows), encoding="utf-8", newline="")
        paths.append(path)
    return paths


def load_responses(xml_path: str) -> Dict[str, str]:
    data = Path(xml_path).read_text(encoding="utf-8")
    out = {}
    for block in re.findall(r"<Question>(.*?)</Question>", data, re.S):
        ident, response = re.search(r"<ID>(.*?)</ID>", block, re.S), re.search(r"<Response>(.*?)</Response>", block, re.S)
        if ident and response:
            out[ident.group(1).strip()] = response.group(1)
    return out


def read_tsv(path: str) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    return [dict(zip(rows[0], row)) for row in rows[1:]] if rows else []


def run_on_texts(pipeline, texts: Dict[str, str], out_dir: str = "outputs") -> Dict[str, dict]:
    acc: Tuple[List[Row], List[Row], List[Row]] = ([], [], [])
    results = {}
    for qid, text in texts.items():
        result = pipeline.analyze(text)
        results[qid] = result
        for target, part in zip(acc, competition_rows(result, qid)):
            target.extend(part)
    write_competition_files(acc, out_dir)
    (Path(out_dir) / "full_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main(argv: Sequence[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    from verifier import IslamicContentVerifier

    results = run_on_texts(IslamicContentVerifier(), load_responses(argv[1]), argv[2] if len(argv) > 2 else "outputs")
    print(f"{len(results)} responses written to {argv[2] if len(argv) > 2 else 'outputs'}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
