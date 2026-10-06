from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
LABEL2ID = {"O": 0, "B-Ayah": 1, "I-Ayah": 2, "B-Hadith": 3, "I-Hadith": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

DEFAULT_DATASET = BASE_DIR / "data" / "islamic_unified_dataset.jsonl"
DEFAULT_MODEL = "aubmindlab/bert-base-arabertv2"


def load_records(path: Path = DEFAULT_DATASET) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def detection_items(records: Sequence[dict], use_b: bool = True, exclude_ids: Sequence[str] = ()) -> List[Tuple[str, str, List[dict]]]:
    keep = {"real_A", "synthetic"} | ({"real_B"} if use_b else set())
    excluded = set(exclude_ids)
    return [
        (r["id"], r["text"], [{"label": s["label"], "start": s["start"], "end": s["end"]} for s in r["spans"]])
        for r in records
        if r["source"] in keep and r["complete"]["detection"] and r["id"] not in excluded
    ]


def build_character_labels(text: str, annotations: Sequence[dict]) -> List[str]:
    labels = ["O"] * len(text)
    for ann in sorted(annotations, key=lambda a: a["end"] - a["start"], reverse=True):
        start, end = max(0, ann["start"]), min(ann["end"], len(text))
        if start >= end:
            continue
        labels[start] = f"B-{ann['label']}"
        for i in range(start + 1, end):
            labels[i] = f"I-{ann['label']}"
    return labels


def char_labels_to_token_labels(offsets: Sequence[Tuple[int, int]], char_labels: Sequence[str]) -> List[int]:
    return [-100 if start == end else LABEL2ID[char_labels[start]] for start, end in offsets]


def train(records: Sequence[dict], out_dir: str, model_name: str = DEFAULT_MODEL, epochs: int = 8, batch: int = 4,
          lr: float = 3e-5, use_b: bool = True, exclude_ids: Sequence[str] = (), seed: int = 42,
          max_length: int = 512, stride: int = 256) -> str:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = detection_items(records, use_b, exclude_ids)
    print(f"Training on {len(items)} texts with {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class Windows(Dataset):

        def __init__(self) -> None:
            self.windows: List[Dict[str, "torch.Tensor"]] = []
            for _, text, annotations in items:
                char_labels = build_character_labels(text, annotations)
                encoded = tokenizer(text, return_offsets_mapping=True, truncation=False)
                ids, mask, offsets = encoded["input_ids"], encoded["attention_mask"], encoded["offset_mapping"]
                start = 0
                while start < len(ids):
                    end = min(start + max_length, len(ids))
                    pad = max_length - (end - start)
                    window_offsets = offsets[start:end] + [(0, 0)] * pad
                    self.windows.append({
                        "input_ids": torch.tensor(ids[start:end] + [tokenizer.pad_token_id] * pad),
                        "attention_mask": torch.tensor(mask[start:end] + [0] * pad),
                        "labels": torch.tensor(char_labels_to_token_labels(window_offsets, char_labels)),
                    })
                    if end == len(ids):
                        break
                    start += stride

        def __len__(self) -> int:
            return len(self.windows)

        def __getitem__(self, index: int):
            return self.windows[index]

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)
    loader = DataLoader(Windows(), batch_size=batch, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for batch_data in loader:
            optimizer.zero_grad()
            loss = model(**{k: v.to(device) for k, v in batch_data.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        print(f"  epoch {epoch + 1}/{epochs}  loss {running / len(loader):.4f}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Saved to", out_dir)
    return out_dir


def evaluate_detector(detector, records: Sequence[dict], ids: Sequence[str]) -> float:
    sys.path.insert(0, str(BASE_DIR))
    from evaluate import detection_score

    selected = [r for r in records if r["id"] in set(ids)]
    texts = {r["id"]: r["text"] for r in selected}
    gold = {r["id"]: [{k: s[k] for k in ("label", "start", "end")} for s in r["spans"]] for r in selected}
    predicted = {qid: [{"label": d.label, "start": d.start, "end": d.end} for d in detector.detect(text)] for qid, text in texts.items()}
    return detection_score(texts, gold, predicted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--out", default="models/span_detector")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--holdout", type=int, default=0, help="hide the first N real_A responses and report an honest F1 on them")
    args = parser.parse_args()

    records = load_records(Path(args.dataset))
    holdout = [r["id"] for r in records if r["source"] == "real_A"][: args.holdout]
    train(records, args.out, args.model, args.epochs, args.batch, args.lr, exclude_ids=holdout)
    if holdout:
        sys.path.insert(0, str(BASE_DIR))
        from detector import RuleDetector

        print(f"Rule-based F1 on the held-out responses: {evaluate_detector(RuleDetector(None), records, holdout):.4f}")
        print("Load the trained model yourself to score it; it is not part of the production application.")


if __name__ == "__main__":
    main()
