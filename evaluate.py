from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from retrieval import BASE_DIR, DATA_DIR
from verifier import IslamicContentVerifier

DEV_SUBSET_PATH = BASE_DIR / "data" / "islamiceval_dev_subset.jsonl"
NO_SOURCE = "خطأ"
_TAGS = {"Ayah": 1, "Hadith": 2}


def load_dev_subset(path: Path = DEV_SUBSET_PATH) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]



def _macro_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for label in sorted(set(truth.tolist()) | set(pred.tolist())):
        tp = int(((truth == label) & (pred == label)).sum())
        fp = int(((truth != label) & (pred == label)).sum())
        fn = int(((truth == label) & (pred != label)).sum())
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def detection_score(texts: Dict[str, str], gold: Dict[str, list], predicted: Dict[str, list]) -> float:
    total = 0.0
    for qid, gold_spans in gold.items():
        pred_spans = predicted.get(qid, [])
        if not gold_spans:
            total += 1.0 if not pred_spans else 0.0
            continue
        if not pred_spans:
            continue
        length = len(texts[qid])
        truth, pred = np.zeros(length, dtype=int), np.zeros(length, dtype=int)
        for span in gold_spans:
            truth[span["start"]:span["end"]] = _TAGS[span["label"]]
        for span in pred_spans:
            pred[span["start"]:span["end"]] = _TAGS[span["label"]]
        total += _macro_f1(truth, pred)
    return total / max(len(gold), 1)


_DIAC_REPLACEMENTS = [("َا", "ا"), ("ِي", "ي"), ("ُو", "و"), ("الْ", "ال"), ("ْ", ""), ("اَ", "ا"), ("اِ", "ا"),
                      ("لِا", "لا"), ("اً", "ًا")]


def _remove_default_diacritics(text: str) -> str:
    for old, new in _DIAC_REPLACEMENTS:
        text = text.replace(old, new)
    return text


class CorrectionScorer:

    def __init__(self, quran_path: Path = DATA_DIR / "quran.json", hadith_path: Path = DATA_DIR / "hadith.json") -> None:
        with open(quran_path, encoding="utf-8") as handle:
            quran = json.load(handle)
        with open(hadith_path, encoding="utf-8") as handle:
            hadith = json.load(handle)
        self.database = {_remove_default_diacritics(a["ayah_text"]) for a in quran if a and a.get("ayah_text")}
        for entry in hadith:
            for key in ("hadithTxt", "Matn"):
                if entry and entry.get(key):
                    self.database.add(_remove_default_diacritics(entry[key]))

    def is_correct(self, predicted: str, reference: str) -> bool:
        pred, ref = _remove_default_diacritics(predicted), _remove_default_diacritics(reference)
        return pred == ref or (ref in pred and pred in self.database)


def evaluate_detection(pipeline: IslamicContentVerifier, records: List[dict]) -> float:
    real_a = [r for r in records if r["source"] == "real_A"]
    texts = {r["id"]: r["text"] for r in real_a}
    gold = {r["id"]: [{k: s[k] for k in ("label", "start", "end")} for s in r["spans"]] for r in real_a}
    predicted = {
        r["id"]: [{"label": d.label, "start": d.start, "end": d.end} for d in pipeline.detector.detect(r["text"])]
        for r in real_a
    }
    return detection_score(texts, gold, predicted)


def evaluate_verification(pipeline: IslamicContentVerifier, records: List[dict]) -> dict:
    items = [(r["text"], s) for r in records if r["source"] == "real_B" for s in r["spans"]]
    correct = verified_when_wrong = wrong_total = verified_when_right = right_total = 0
    for text, span in items:
        quote = text[span["start"]:span["end"]].strip()
        gold = span["verdict"]
        predicted = pipeline.verifier.verify(quote, span["label"]).verdict
        correct += predicted == gold

        status = pipeline.analyze_spans(text, [span])["spans"][0]["status"]
        if gold == "Incorrect":
            wrong_total += 1
            verified_when_wrong += status == "VERIFIED"
        else:
            right_total += 1
            verified_when_right += status == "VERIFIED"
    return {
        "n_spans": len(items),
        "verdict_accuracy": correct / len(items),
        "all_correct_baseline": sum(s["verdict"] == "Correct" for _, s in items) / len(items),
        "pipeline_incorrect_marked_verified": f"{verified_when_wrong}/{wrong_total}",
        "pipeline_correct_marked_verified": f"{verified_when_right}/{right_total}",
    }


def evaluate_correction(pipeline: IslamicContentVerifier, records: List[dict]) -> dict:
    scorer = CorrectionScorer()
    corr_cfg = pipeline.cfg.corrector
    items = [(r["text"], s) for r in records if r["source"] == "real_C" for s in r["spans"]]
    correct = 0
    for text, span in items:
        quote = text[span["start"]:span["end"]]
        match = pipeline.corrector.match(quote, span["label"])
        strong = corr_cfg.quran_strong if span["label"] == "Ayah" else corr_cfg.hadith_strong
        proposed = match.text if (match and match.strength >= strong
                                  and (span["label"] != "Ayah" or match.full_ratio >= corr_cfg.min_full_ratio)) else NO_SOURCE
        correct += scorer.is_correct(proposed, span["correction"])
    return {
        "n_spans": len(items),
        "correction_accuracy": correct / len(items),
        "all_no_source_baseline": sum(s["correction"] == NO_SOURCE for _, s in items) / len(items),
    }


_CARRIERS = [
    "ومما يحسن التنبيه عليه في هذا الباب ما يلي {q} وهذا ما أردنا بيانه هنا.",
    "وقد تأمل الناس هذا المعنى طويلا. {q} ثم عادوا إلى أعمالهم.",
    "يقول المتحدث في ختام كلمته {q} وبعدها انصرف الحاضرون.",
]
_PROSE = [
    "الصبر مفتاح الفرج، والإنسان الذي يتحلى بالصبر يستطيع أن يواجه مشكلات الحياة بهدوء وثبات وعزيمة.",
    "تعلن الجامعة عن فتح باب التسجيل للفصل الدراسي القادم، ويمكن للطلاب مراجعة الموقع الإلكتروني لمعرفة المواعيد.",
    "ارتفعت أسعار القمح في الأسواق العالمية هذا الأسبوع بسبب قلة الإمدادات وزيادة الطلب من الدول المستوردة.",
    "من المهم أن نتعاون جميعا في بناء مجتمع قوي يحترم العلم والعمل ويقدر جهود المخلصين من أبنائه.",
    "ذهبت إلى السوق صباحا واشتريت الخضروات والفواكه ثم عدت إلى البيت وأعددت الغداء للأسرة كلها.",
    "يحتاج المشروع إلى خطة واضحة وميزانية مدروسة وفريق متعاون حتى ينجح في الوصول إلى أهدافه المرسومة.",
]


def evaluate_unannounced(pipeline: IslamicContentVerifier, records: List[dict]) -> dict:
    """Recall on real quotations embedded in plain prose WITHOUT quotation marks or introductory phrases, plus false
    alarms on ordinary prose. Quotations of 6+ words come from the IslamicEval 1A development responses."""
    quotes = [(r["text"][s["start"]:s["end"]].strip(), s["label"]) for r in records if r["source"] == "real_A"
              for s in r["spans"] if len(r["text"][s["start"]:s["end"]].split()) >= 6 and "\n" not in r["text"][s["start"]:s["end"]]]
    hit_exact = hit_altered = total_altered = genuine = genuine_hit = 0
    for i, (quote, label) in enumerate(quotes):
        text = _CARRIERS[i % len(_CARRIERS)].format(q=quote)
        start = text.index(quote)
        found = [d for d in pipeline.detector.detect(text) if d.start < start + len(quote) and d.end > start]
        hit_exact += any(d.label == label for d in found)

        status = pipeline.analyze_spans(text, [{"label": label, "start": start, "end": start + len(quote)}])["spans"][0]["status"]
        if status in ("VERIFIED", "CORRECTED"):
            genuine += 1
            genuine_hit += any(d.label == label for d in found)
        words = quote.split()   
        if len(words) >= 8:
            altered = words[:]
            altered[len(words) // 2] = words[(len(words) // 2 + 3) % len(words)]
            altered_quote = " ".join(altered)
            text = _CARRIERS[i % len(_CARRIERS)].format(q=altered_quote)
            total_altered += 1
            hit_altered += any(d.label == label for d in pipeline.detector.detect(text))
    false_alarms = sum(len(pipeline.detector.detect(p)) for p in _PROSE)
    return {
        "quotations_tested": len(quotes), "found_unmarked_recall": round(hit_exact / max(len(quotes), 1), 4),
        "corpus_backed_tested": genuine, "found_unmarked_recall_corpus_backed": round(genuine_hit / max(genuine, 1), 4),
        "altered_tested": total_altered, "found_with_one_changed_word_recall": round(hit_altered / max(total_altered, 1), 4),
        "prose_sentences": len(_PROSE), "false_alarms_on_prose": false_alarms,
    }


def run_evaluation() -> dict:
    records = load_dev_subset()
    pipeline = IslamicContentVerifier()
    rules_only = IslamicContentVerifier(retriever=pipeline.retriever, use_scanner=False)
    return {
        "dataset": "IslamicEval 2025 development subset (50 responses per subtask)",
        "detector": pipeline.detector_name,
        "1A_detection_macro_f1_hybrid": evaluate_detection(pipeline, records),
        "1A_detection_macro_f1_rules_only": evaluate_detection(rules_only, records),
        "unannounced_quotations": evaluate_unannounced(pipeline, records),
        "1B_verification": evaluate_verification(pipeline, records),
        "1C_correction": evaluate_correction(pipeline, records),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="optional path to write the results as JSON")
    args = parser.parse_args()
    results = run_evaluation()
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
