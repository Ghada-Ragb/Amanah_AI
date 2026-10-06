# Amanah AI — Full Documentation

*AI-Powered Verification and Correction of Quranic and Prophetic Quotations* · Arabic version: [DOCUMENTATION_AR.md](DOCUMENTATION_AR.md)

## 1. Problem

Large language models produce Quranic verses and Hadith that look right and are often wrong (substituted word, spliced verses, invented hadith,
wrong speaker). In religious content a confident error is harmful and a missing answer is acceptable. The task (IslamicEval 2025, Subtask 1) is
to **detect** quotations (1A), **verify** them (1B) and **correct** the wrong ones from the official sources (1C).

## 2. Solution overview

A retrieval-and-alignment engine that treats the reference corpora as the only authority:

* the model (or any author) may write the text, but **only the corpora decide** whether a quotation is right;
* corrections are **copied** from the source, never generated;
* when evidence is weak the system **abstains** (human review).

Two pages: **Ask & Verify** (default) and **Verify directly**; both have «جرّب مثالًا».

## 3. Workflow

1. **Input** – a question (Ask & Verify) or a text (Verify directly). For a question, a language model answers through a proxy; the answer is sanitised (foreign scripts removed).
2. **Detection (1A)** – candidate spans come from quotation marks and brackets, from the corpus scanner (for unannounced quotations) and, if configured, from a CAMeLBERT-MSA token classifier. Overlaps are resolved (the longer / corpus-backed span wins).
3. **Typing** – each span is typed *Ayah* or *Hadith* by word coverage against the corpora. Introductory phrases («قال الله تعالى», «قال رسول الله ﷺ») are only hints; a disagreement between phrase and corpus is reported as a note (`misattributed`, `is_ayah`, `is_hadith`).
4. **Retrieval** – BM25 candidates, re-ranked by character n-gram similarity.
5. **Alignment** – word alignment against the best candidate (phonetic skeleton, sliding window, LCS, dynamic gap) → exact / substring / altered / unrelated.
6. **Decision (1B)** – `VERIFIED`, `CORRECTED`, `UNSUPPORTED` or `HUMAN_REVIEW` (see §6).
7. **Grounding guard** – every correction and suggestion is checked against the cited record (§7).
8. **Report** – coloured cards with evidence, the correction (Quran only) and the reason in plain Arabic; one-click copy of the corrected text.

## 4. Components

| Module | Responsibility |
|---|---|
| `normalization.py`, `idgham.py` | Arabic normalisation (diacritics, alef/ya/ta marbuta folding, phonetic skeleton); mushaf rendering of corrected text |
| `index_builder.py`, `retrieval.py` | Builds and loads `index/quran.idx.gz` (2.1 MB) and `index/hadith.idx.gz` (8.2 MB); BM25 + n-gram retriever, lazy Hadith loading |
| `detector.py` | Quotation-mark/bracket segments, corpus-first typing, multi-ayah windows |
| `scanner.py` | Corpus scanner: whole-ayah index, phonetic n-gram anchors (Quran), BM25 windows (Hadith); hybrid overlap resolution |
| `alignment.py`, `similarity.py` | Alignment and similarity signals |
| `verifier.py` | Decision logic, grounding guard, final verdict |
| `camelbert_adapter.py` | Optional hosted CAMeLBERT-MSA spans, silent merge (corpus-backed detector wins overlaps; failures ignored) |
| `llm_client.py`, `server/ask-core.js` | Question answering; identical Arabic-only system prompt in both; answer sanitiser |
| `ui.py`, `app.py`, `build_static_space.py` | Arabic UI, local Gradio app, static site (Pyodide in a Web Worker) |

## 5. AI, RAG and verification methodology

* **Retrieval-augmented verification.** The "retrieval" is BM25 over pre-tokenised Quran ayahs (6,236) and Hadith records (34,994 records from six books).
  The retrieved record is not used to *generate* text; it is used to *judge* and to *copy* from.
* **LLM role.** The language model only produces the answer to be checked (Ask & Verify). Verification never asks a model to decide; this keeps decisions reproducible and testable.
* **Optional learned detector.** `research/train_detector.py` fine-tunes CAMeLBERT-MSA for 1A (labels `O, B/I-Ayah, B/I-Hadith`). If a model is published and set in `config.json` (`hf.model`), its spans are merged silently; without it nothing changes. No weights are bundled.
* **Normalisation.** Quran matching ignores diacritics and spelling conventions (alef forms, ta marbuta, alef maqsura) but keeps phonetic skeletons; Hadith matching uses lenient normalisation.

## 6. Decision rules

| Status | When | Correction |
|---|---|---|
| `VERIFIED` | Exactly equals its source after normalisation (whole ayah, or exact substring of a record) | – |
| `CORRECTED` | A Quran passage whose best source is strong and local, but words differ | Verbatim source text |
| `UNSUPPORTED` | Nothing in the corpora resembles it | none, and no evidence is shown |
| `HUMAN_REVIEW` | Weak/ambiguous evidence, a Hadith that differs in any word, a quote under 3 words, or an ungrounded result | none |

Details: Hadith are verified only on exact word match; any added, missing or changed word is `hadith_altered`. A quote of fewer than three words from the rules is `too_short`
unless it equals a whole ayah. The 1B verdict is *Correct* only for `VERIFIED`.

## 7. Safety and abstention

* No fabricated verse, hadith, reference, source, similarity or correction. `_ground()` verifies that the cited source exists and that any correction/suggestion is a normalised substring of the cited record; otherwise it removes it and sets `HUMAN_REVIEW` (`ungrounded`).
* The grounding result is stored in each span (`grounding`) and shown in tests.
* Human review is a first-class outcome, not an error.
* Model answers are stripped of non-Arabic scripts; the proxy limits size, origin and rate; no secret lives in the repository or page.
* The tool checks *text against sources*; it does not rule on authenticity, grading or jurisprudence.

## 8. Data and sources

See [DATA_SOURCES_AND_LICENSES.md](DATA_SOURCES_AND_LICENSES.md) and [REFERENCES.md](REFERENCES.md). Corpora: `data/quran.json` (Tanzil text, from the IslamicEval 2025 repository), `data/hadith.json` (six Hadith books, same repository), IslamicEval 2025 development subset for evaluation; all verified identical to the upstream files.

## 9. Testing and evaluation

* `python -m unittest discover -s tests` — 69 tests: normalisation, detection, alignment, decisions, grounding, Hadith strictness, examples (each of the 25 examples reproduces its stored statuses and renders), proxy prompt parity, Colab notebook in sync with the repository files, loading indicator, secret scan.
* `python evaluate.py` — IslamicEval development subset: 1A 0.8274 (rules only 0.8379), 1B 93.12%, 1C 74.30% (tuned on the same data; optimistic).
* `python research/stress_test.py 60 7` — synthetic stress test; results in the README. No fabricated or spliced text was marked verified.
* UI: the static page was exercised in a browser (desktop and phone widths) with a stubbed worker; the Python core was run under Pyodide 0.26.4 in Node.

## 10. Performance

Findings and measurements are in the README (*Performance*). Root cause of slow loading: duplicate downloads and a large Hadith index. Final CPU timings: Pyodide ready 1.6–2.2 s, Quran pipeline 2.1–2.8 s, Hadith loaded 2.6–3.4 s, three verifications done by 4.5–5.0 s. Network time was not measurable from the authoring environment.

## 11. Deployment

Static site (`python build_static_space.py` → `static_space/`) on a Hugging Face Static Space or Cloudflare Pages, plus a Cloudflare Worker for Ask & Verify. See [DEPLOYMENT.md](DEPLOYMENT.md).

## 12. Usage

* **Ask & Verify:** type a question → *اسأل ثم تحقّق*; or press *جرّب مثالًا* to verify a saved answer.
* **Verify directly:** paste text → *تحقّق من النص*; or *جرّب مثالًا*.
* **Reading the result:** green = verified; red = mismatch (Quran: the exact source text and a copy button); amber = a person must decide.
* **Colab:** `colab_verification_pipeline/` (notebook + `RUN_ON_COLAB.txt`).

## 13. Limitations

* Corpus matching, not understanding: very short or far-from-corpus quotations may be missed; unmarked Hadith were found in 73% of stress-test cases.
* Hadith are never auto-corrected and authenticity is out of scope.
* The in-browser build was not run end to end in the authoring environment; the optional CAMeLBERT model is not bundled.
* The Hadith compilation's own licence is not stated upstream; the Quran text must stay unchanged (Tanzil, CC BY 3.0).
* Metrics come from development data on which thresholds were tuned.
