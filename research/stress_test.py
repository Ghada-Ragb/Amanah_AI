from __future__ import annotations

import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from normalization import normalize_for_matching  
from verifier import IslamicContentVerifier  

DIAC = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
INTROS_Q = ["قال الله تعالى: ", "وفي القرآن الكريم: ", "ويقول ربنا سبحانه: ", "جاء في كتاب الله: "]
INTROS_H = ["قال رسول الله ﷺ: ", "وفي الحديث الشريف: ", "روى البخاري أن النبي ﷺ قال: ", "وقال النبي ﷺ: "]
TAIL = ["، وهذا واضح.", ". وبالله التوفيق.", "، فتأمل ذلك."]


def strip(text: str) -> str:
    return re.sub(r"\s+", " ", DIAC.sub("", text)).strip()


def build_cases(kb, n: int, rng: random.Random):
    cases = []
    quran = [r for r in kb.quran if len(r["text"].split()) >= 4]
    vocab = sorted({w for r in kb.quran for w in strip(r["text"]).split()})
    hadith = [r for r in kb.hadith if r["matn"] and len(r["matn"].split()) >= 8]

    def wrap(kind, label, quote, expect, source=None, mode="quoted", gold=None):
        intro = rng.choice(INTROS_Q if label == "Ayah" else INTROS_H)
        if mode == "quoted":
            text, q = f"{intro}\"{quote}\"{rng.choice(TAIL)}", quote
        elif mode == "bare":
            text, q = f"{intro}{quote}{rng.choice(TAIL)}", quote
        else:   
            text, q = f"ونختم هذا الحديث بما يلي {quote} والله أعلم.", quote
        start = text.index(q)
        cases.append({"kind": kind, "label": label, "text": text, "span": (start, start + len(q)), "expect": expect,
                      "gold": gold, "mode": mode})

    for rec in rng.sample(quran, n):
        words = rec["text"].split()
        plain = strip(rec["text"]).split()
        wrap("quran_exact", "Ayah", rec["text"], "VERIFIED")
        wrap("quran_plain", "Ayah", " ".join(plain), "VERIFIED")
        wrap("quran_unmarked", "Ayah", " ".join(plain), "VERIFIED", mode="bare")
        wrap("quran_embedded", "Ayah", " ".join(plain), "VERIFIED", mode="embedded")
        if len(plain) >= 8:
            a = rng.randint(0, len(plain) - 6)
            wrap("quran_partial", "Ayah", " ".join(plain[a:a + rng.randint(5, min(8, len(plain) - a))]), "VERIFIED")
        if len(plain) >= 5:
            i = rng.randrange(1, len(plain) - 1)
            alt = plain[:i] + [rng.choice([w for w in vocab if w != plain[i]])] + plain[i + 1:]
            wrap("quran_substituted", "Ayah", " ".join(alt), "CORRECTED", gold=" ".join(plain))
            alt = plain[:i] + plain[i + 1:]
            wrap("quran_dropped", "Ayah", " ".join(alt), "CORRECTED", gold=" ".join(plain))
            j = rng.randrange(0, len(plain) - 1)
            sw = list(plain); sw[j], sw[j + 1] = sw[j + 1], sw[j]
            if sw != plain:
                wrap("quran_swapped", "Ayah", " ".join(sw), "CORRECTED", gold=" ".join(plain))
        fake = [rng.choice(vocab) for _ in range(rng.randint(6, 10))]
        wrap("quran_fabricated", "Ayah", " ".join(fake), "NOT_VERIFIED")
    for rec in rng.sample(hadith, n):
        words = strip(rec["matn"]).split()
        k = rng.randint(7, min(24, len(words)))
        a = rng.randint(0, len(words) - k)
        frag = words[a:a + k]
        raw = rec["matn"].split()
        rawfrag = raw[a:a + k] if len(raw) == len(words) else frag
        wrap("hadith_exact", "Hadith", " ".join(rawfrag), "VERIFIED")
        wrap("hadith_plain", "Hadith", " ".join(frag), "VERIFIED")
        wrap("hadith_unmarked", "Hadith", " ".join(frag), "VERIFIED", mode="bare")
        other = rng.choice(hadith)
        ow = strip(other["matn"]).split()
        half = frag[: k // 2] + ow[: max(4, k // 2)]
        wrap("hadith_spliced", "Hadith", " ".join(half), "NOT_VERIFIED")
        fake = [rng.choice(vocab) for _ in range(rng.randint(8, 12))]
        wrap("hadith_fabricated", "Hadith", " ".join(fake), "NOT_VERIFIED")
    return cases


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) / max(1, b[1] - b[0])


def evaluate(cases, pipe, show=15):
    stats = defaultdict(lambda: defaultdict(int))
    failures = defaultdict(list)
    for c in cases:
        result = pipe.analyze(c["text"])
        best = max(result["spans"], key=lambda s: overlap((s["start"], s["end"]), c["span"]), default=None)
        s = stats[c["kind"]]
        s["n"] += 1
        found = best is not None and overlap((best["start"], best["end"]), c["span"]) >= 0.6
        fabricated = c["expect"] == "NOT_VERIFIED"
        if found:
            s["found"] += 1
            s["typed"] += best["type"] == c["label"]
        status = best["status"] if found else "MISSED"
        if fabricated:
            ok = status != "VERIFIED"      
            s["ok"] += ok
            s["false_verified"] += status == "VERIFIED"
        elif c["expect"] == "VERIFIED":
            ok = status == "VERIFIED"
            s["ok"] += ok
        else:
            ok = status in ("CORRECTED",)
            s["ok"] += ok
            s["flagged"] += status in ("CORRECTED", "HUMAN_REVIEW", "UNSUPPORTED")
            if status == "CORRECTED":
                fixed = strip(best["correction"]["display_text"]) if best["correction"] else ""
                s["fix_exact"] += normalize_for_matching(fixed) == normalize_for_matching(c["gold"])
        if not ok and len(failures[c["kind"]]) < show:
            failures[c["kind"]].append((status, c["text"][:150]))
    return stats, failures


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 60
    seed = int(argv[2]) if len(argv) > 2 else 7
    pipe = IslamicContentVerifier()
    cases = build_cases(pipe.retriever, n, random.Random(seed))
    stats, failures = evaluate(cases, pipe)
    print(f"{'kind':20s} {'n':>4s} {'found':>6s} {'typed':>6s} {'ok':>6s} extra")
    for kind, s in sorted(stats.items()):
        extra = ""
        if "fix_exact" in s: extra = f"flagged={s['flagged']/s['n']:.2f} fix_exact={s['fix_exact']/s['n']:.2f}"
        if "false_verified" in s: extra = f"false_verified={s['false_verified']}"
        print(f"{kind:20s} {s['n']:4d} {s['found']/s['n']:6.2f} {s['typed']/max(1,s['found']):6.2f} {s['ok']/s['n']:6.2f} {extra}")
    if "--fail" in argv:
        for kind, rows in failures.items():
            print("\n==", kind)
            for status, text in rows:
                print(" ", status, "|", text)


if __name__ == "__main__":
    main(sys.argv)
