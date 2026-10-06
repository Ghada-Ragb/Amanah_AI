from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "amanah_ai_verification_pipeline.ipynb"
MODULES = [
    ("normalization.py", "Arabic normalisation: phonetic skeletons, diacritic handling and tokenisation used by every other stage."),
    ("idgham.py", "Mushaf-style rendering (idgham) so that corrections taken from the Quran look like the printed text."),
    ("alignment.py", "Word-level alignment between a quotation and a source: sliding window, longest common subsequence and a dynamic gap."),
    ("similarity.py", "Similarity indicators (coverage, token overlap, edit similarity ...) computed on the matched region only."),
    ("index_builder.py", "Search indexes: BM25 postings and phonetic n-gram anchors, built once from the corpora."),
    ("retrieval.py", "Retriever: BM25 recall and character n-gram re-ranking over the Quran and Hadith indexes."),
    ("detector.py", "Rule-based detector: quotation marks and brackets, typed by the corpora (introductory phrases are only hints)."),
    ("scanner.py", "Corpus scanner for unannounced quotations, whole-ayah matching and the hybrid detector."),
    ("verifier.py", "The decision engine: verification, source-backed correction, abstention and the anti-hallucination guard."),
    ("llm_client.py", "Optional LLM client for the Ask & Verify mode, including the foreign-script sanitiser."),
    ("benchmark_format.py", "IslamicEval 2025 Subtask 1 output formats (1A / 1B / 1C)."),
    ("ui.py", "Arabic HTML report renderer (used here to display results inside the notebook)."),
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build() -> dict:
    cells = [
        md("# Amanah AI — Verification Pipeline\n"
           "**AI-Powered Verification and Correction of Quranic and Prophetic Quotations**\n\n"
           "This notebook contains the complete Python verification pipeline of the project: detection, retrieval (RAG-style evidence "
           "lookup), word-level alignment, decision and source-backed correction. It never generates scripture: every reference and "
           "correction is copied verbatim from the bundled corpora, otherwise the result is *Incorrect* or *Needs Human Review*."),
        md("## 1. Configuration\nChoose where the corpora come from and whether the optional LLM step is enabled."),
        code('''
import os, sys, json, pathlib

WORKDIR = (pathlib.Path("/content/amanah_ai") if pathlib.Path("/content").exists() else pathlib.Path("amanah_ai_workdir")).resolve()
WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR))
for folder in ("index", "data", "demo"):
    (WORKDIR / folder).mkdir(exist_ok=True)

USE_LLM = False          # True enables the optional "Ask & Verify" cell (needs a key in Colab Secrets, see the .txt guide)
print("Working directory:", WORKDIR)
'''),
        md("## 2. Provide the corpora\nUpload `index/quran.idx.gz` and `index/hadith.idx.gz` from the repository (fast), or `data/quran.json` and "
           "`data/hadith.json` (the indexes are then built here). Skipped automatically when the files already exist."),
        code('''
have_index = all((WORKDIR / "index" / n).exists() for n in ("quran.idx.gz", "hadith.idx.gz"))
have_json = all((WORKDIR / "data" / n).exists() for n in ("quran.json", "hadith.json"))
if not (have_index or have_json):
    try:
        from google.colab import files
        print("Select the two index files (or the two JSON corpora) from the repository.")
        for name, payload in files.upload().items():
            target = "index" if name.endswith(".idx.gz") else "data"
            (WORKDIR / target / name).write_bytes(payload)
    except ImportError:
        raise SystemExit("Place index/*.idx.gz or data/*.json in the working directory, then re-run this cell.")
print({p.name: p.stat().st_size for folder in ("index", "data") for p in (WORKDIR / folder).glob("*")})
'''),
    ]
    for name, blurb in MODULES:
        source = (ROOT / name).read_text(encoding="utf-8")
        cells.append(md(f"### Module: `{name}`\n{blurb}"))
        cells.append(code(f"%%writefile {name}\n{source}"))
    cells += [
        md("## 3. Build the indexes if needed\nOnly runs when the JSON corpora were provided instead of the prebuilt indexes."),
        code('''
if not all((WORKDIR / "index" / n).exists() for n in ("quran.idx.gz", "hadith.idx.gz")):
    from index_builder import build_quran_index, build_hadith_index, save_index
    save_index(build_quran_index(WORKDIR / "data" / "quran.json"), WORKDIR / "index" / "quran.idx.gz")
    save_index(build_hadith_index(WORKDIR / "data" / "hadith.json"), WORKDIR / "index" / "hadith.idx.gz")
print("Indexes ready:", sorted(p.name for p in (WORKDIR / "index").glob("*")))
'''),
        md("## 4. Load the pipeline\nCreates the verifier (Quran index first, Hadith index on first use) and warms the Hadith index."),
        code('''
import time
from verifier import IslamicContentVerifier

started = time.time()
pipeline = IslamicContentVerifier()
pipeline.retriever.warm()
print(f"Pipeline ready in {time.time() - started:.1f}s")
'''),
        md("## 5. Verify a text\nPaste any Arabic text. Each Quran/Hadith quotation is detected, compared word by word with its source, and marked "
           "*verified*, *incorrect* (with the exact source text as correction) or *needs human review*."),
        code('''
from IPython.display import HTML, display
import ui

def verify(text, show_html=True):
    result = pipeline.analyze(text)
    if show_html:
        display(HTML("<style>" + ui.CSS + "</style>" + ui.render_results(result)))
    return result

sample = 'قال الله تعالى: "إِنَّ مَعَ الْعُسْرِ يُسْرًا". وقال رسول الله ﷺ: "الطُّهُورُ شَطْرُ الْإِيمَانِ" رواه مسلم.'
result = verify(sample)
'''),
        md("## 6. Compact decision table\nA plain-text view of every decision, useful for logs and for checking the abstention behaviour."),
        code('''
def table(result):
    rows = [(s["id"], s["type"], s["status"], s["reason"]["code"], s["grounding"], s["text"][:48]) for s in result["spans"]]
    print(f"{'#':>2}  {'type':6} {'status':13} {'reason':22} {'grounding':10} text")
    for r in rows:
        print(f"{r[0]:>2}  {r[1]:6} {r[2]:13} {r[3]:22} {r[4]:10} {r[5]}")

table(result)
'''),
        md("## 7. Run the bundled examples\nThe examples cover verified, incorrect (with correction), unsupported, uncertain and human-review cases. "
           "The next cell writes them to `demo/examples.json`."),
        code("%%writefile demo/examples.json\n" + (ROOT / "demo" / "examples.json").read_text(encoding="utf-8")),
        md("Loads the saved examples and prints the decision of every quotation in each one."),
        code('''
examples = json.loads((WORKDIR / "demo" / "examples.json").read_text(encoding="utf-8"))
for example in examples:
    print("\\n==", example["id"], "-", example["title"])
    table(pipeline.analyze(example["text"]))
'''),
        md("## 8. Optional: Ask & Verify\nAsks an LLM a question and verifies every quotation in its answer. Disabled unless `USE_LLM = True` and a key "
           "is stored in Colab *Secrets* as `OPENAI_API_KEY` (never paste a key into the notebook)."),
        code('''
if USE_LLM:
    from google.colab import userdata
    from llm_client import LLMSettings, generate, sanitize_answer
    settings = LLMSettings(api_key=userdata.get("OPENAI_API_KEY"), model=os.environ.get("OPENAI_MODEL", ""))
    question = "ما فضل الصبر في القرآن والسنة؟"
    answer = sanitize_answer(generate(settings, question))
    print(answer)
    verify(answer)
else:
    print("USE_LLM is False: skipping the optional LLM step.")
'''),
        md("## 9. Export the official IslamicEval formats\nWrites the three Subtask-1 submission files (1A spans, 1B labels, 1C corrections) for a dictionary of `{question_id: text}`."),
        code('''
from benchmark_format import run_on_texts

texts = {"Q1": sample, **{e["id"]: e["text"] for e in examples[:3]}}
run_on_texts(pipeline, texts, out_dir="outputs")
for name in sorted(p.name for p in pathlib.Path("outputs").glob("*.tsv")):
    print("\\n#", name); print(pathlib.Path("outputs", name).read_text(encoding="utf-8")[:400])
'''),
        md("## 10. Optional stress test\nSamples real ayahs and Hadith from the corpora, builds exact, re-spelled, partial, altered and fabricated variants and "
           "reports detection, verdict and correction rates per kind."),
        code("%%writefile stress_test.py\n" + (ROOT / "research" / "stress_test.py").read_text(encoding="utf-8")),
        md("Runs 20 sampled cases per kind (seed 7) and prints detection, verdict and correction rates; any fabricated text marked verified would be listed."),
        code('''
import random, importlib, stress_test
importlib.reload(stress_test)
cases = stress_test.build_cases(pipeline.retriever, 20, random.Random(7))
stats, failures = stress_test.evaluate(cases, pipeline)
for kind, s in sorted(stats.items()):
    print(f"{kind:20s} n={s['n']:3d} found={s['found']/s['n']:.2f} ok={s['ok']/s['n']:.2f} false_verified={s.get('false_verified', 0)}")
'''),
        md("## 11. Anti-hallucination self-check\nEvery reference and correction in a report must be found word for word inside the cited record. The check below fails loudly otherwise."),
        code('''
checked = 0
for example in examples:
    for span in pipeline.analyze(example["text"])["spans"]:
        assert span["grounding"] == "verified", (example["id"], span["status"])
        checked += 1
print(f"All {checked} reports are grounded in the bundled corpora.")
'''),
    ]
    return {"cells": cells, "metadata": {"colab": {"name": "amanah_ai_verification_pipeline.ipynb", "provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    OUT.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
