# Amanah AI

**AI-Powered Verification and Correction of Quranic and Prophetic Quotations**
(Arabic tagline: *نظام ذكي للتحقق من الاقتباسات القرآنية والحديثية*) · Arabic version: [README_AR.md](README_AR.md)

Amanah AI finds Quran verses and Hadith quotations in any text (typically a language-model answer), compares each one word by word
with the bundled source corpora, **corrects Quranic wording verbatim from the source text**, and sends everything uncertain to
**human review**. It never invents a verse, hadith, reference, similarity or correction: when the evidence is insufficient it abstains.

Built for the *AI in Service of Islamic Content 2026* hackathon (IslamicEval 2025 Subtask 1: detection 1A, verification 1B, correction 1C).



**## Live demo**

\<!-- LIVE_DEPLOYMENT_URL -->
**\*\*Live URL: \`[https://ghada-99-ragab-amanah-ai.static.hf.space]\`\*\*** (replace after deploying; see [docs/DEPLOYMENT.md]\(docs/DEPLOYMENT.md))

**\*\*Demo Video: \`[ADD_DEMO_VIDEO_URL_HERE]\`\*\***

\<!-- /LIVE_DEPLOYMENT_URL -->

## The problem

Language models quote the Quran and the Hadith fluently and often wrongly: a word is replaced, a verse is spliced with another,
a hadith is invented or attributed to the wrong speaker. A reader cannot tell. In religious content a confident mistake is
worse than no answer, so the system must be able to say *"this is wrong"*, *"here is the exact source text"* or *"I am not sure,
a human must decide"* — and never guess.

## The solution

1. **Ask & Verify (default page)** – the user asks a question, a language model answers, and the verification engine checks every quotation in that answer.
2. **Verify directly** – the user pastes any text and gets the same checks.
3. **Three decisions only** – *verified* (green), *mismatch* (red; for Quran the exact source text is offered as correction), *needs human review* (amber, no correction offered).
4. **Try an example** – both pages have a «جرّب مثالًا» button with 25 saved scenarios (all-correct / all-wrong / mixed ayahs and hadiths, one-word changes, short quotes, misattribution, fabricated hadith, …). Each one was run through the engine and its outcome is asserted by a unit test.

## Anti-hallucination design

| Rule | How it is enforced |
|---|---|
| Corrections come only from the source | A correction is the verbatim source text. `verifier._ground()` removes any correction or suggestion that is not a substring of the cited record and downgrades the case to human review (`ungrounded`). |
| Evidence must exist | A cited source must exist in the corpus; an `UNSUPPORTED` span carries no evidence at all. |
| Hadith are never auto-corrected | A Hadith is *verified* only on an exact word match; any added, missing or changed word → human review (`hadith_altered`). A word can change a ruling. |
| Extra word in an ayah ≠ verified | Whole-ayah / substring checks are exact; an altered passage is corrected from the source. |
| Very short quotes | Fewer than 3 words → human review (`too_short`), unless it is exactly a whole ayah. |
| Abstain when unsure | Weak or ambiguous evidence → human review, never a guess. The final 1B verdict is *Correct* only if the status is *verified*. |
| Model output is sanitised | Chinese/Cyrillic/other scripts are stripped from model answers before verification; the model prompt forbids other scripts. |

## Architecture

```
text ─► detection ─► retrieval ─► alignment ─► decision ─► grounding guard ─► report
        │            │            │             │
        │            │            │             └ VERIFIED | CORRECTED | UNSUPPORTED | HUMAN_REVIEW
        │            │            └ phonetic skeleton, sliding window, LCS, dynamic gap
        │            └ BM25 candidates + character n-gram re-rank (Quran ayahs, Hadith records)
        └ quotes / brackets typed by the corpora (intro phrases are only hints)
          + corpus scanner for unannounced quotations (whole-ayah index, n-gram anchors, BM25 windows)
          + optional CAMeLBERT-MSA spans merged silently
```

| File | Role |
|---|---|
| `normalization.py`, `idgham.py` | Phonetic-aware Arabic normalisation; mushaf rendering of corrected text |
| `index_builder.py`, `retrieval.py` | Pre-tokenised indexes (`index/*.idx.gz`) and the lazy retriever |
| `detector.py`, `scanner.py` | Corpus-first quotation typing and unannounced-quotation scanner |
| `alignment.py`, `similarity.py` | Word alignment and similarity signals |
| `verifier.py` | Decision, source-backed correction, grounding guard |
| `camelbert_adapter.py` | Optional hosted CAMeLBERT-MSA spans, merged silently |
| `llm_client.py`, `server/ask-core.js`, `worker/`, `functions/api/ask.js` | Question answering behind a key-less proxy (Cloudflare Workers AI; OpenAI-compatible APIs optional) |
| `ui.py`, `app.py` | Arabic UI and the local Gradio app |
| `build_static_space.py` | Builds the static site; Python runs in the browser through Pyodide |
| `benchmark_format.py`, `evaluate.py`, `research/` | IslamicEval formats, evaluation, stress test, training code |
| `colab_verification_pipeline/` | Colab notebook + `RUN_ON_COLAB.txt` |
| `docs/` | Full documentation, deployment, references (English + Arabic) |

Detailed methodology: [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        
pip install -r requirements.txt
python app.py                                             
python -m unittest discover -s tests                      
python evaluate.py                                        
python research/stress_test.py 60 7                       
python build_static_space.py                              
```

The local app can answer questions only if you set `OPENAI_API_KEY` (copy `.env.example` to `.env`); direct verification needs no key.
The Colab pipeline is documented in [colab_verification_pipeline/RUN_ON_COLAB.txt](colab_verification_pipeline/RUN_ON_COLAB.txt).

## Measured results (prototype, development data)

`python evaluate.py` on the bundled IslamicEval 2025 **development** subset (50 responses per subtask). Thresholds were tuned on these
responses, so the numbers are optimistic and are not a held-out test.

| Component | Result |
|---|---|
| 1A detection, character-level macro F1 (merged detector / rules only) | 0.8274 / 0.8379 |
| 1B verdict accuracy (247 spans; always-"Correct" baseline 59.5%) | 93.12% |
| 1B end-to-end: gold-Incorrect spans marked verified / gold-Correct spans marked verified | 5 of 100 / 124 of 147 |
| 1C correction accuracy (179 spans; always-"no source" baseline 62.0%) | 74.30% |
| Unannounced quotations: recall overall / on corpus-backed ones | 68.1% / 100% |
| False alarms on 6 ordinary prose sentences | 0 |

The cautious design is visible in the 1B end-to-end row: a few correct quotations are sent to review rather than risk accepting a wrong one.
The 1A baseline of 0.9091 in the project proposal comes from earlier CAMeLBERT work, not from this code.

**Synthetic stress test** (`python research/stress_test.py 60 7`, real ayahs/hadiths from the corpora turned into exact, re-spelled, partial, altered and fabricated variants):

| Kind | n | Found | Typed | OK | Extra |
|---|---|---|---|---|---|
| quran exact / plain / embedded / unmarked | 60 each | 1.00 | 1.00 | 1.00 | |
| quran substituted / dropped word | 52 each | 1.00 | 1.00 | 1.00 | flagged 1.00, correction exact 1.00 |
| quran swapped order | 52 | 1.00 | 1.00 | 0.98 | flagged 1.00, correction exact 0.90 |
| quran fabricated | 60 | 0.78 | 1.00 | 1.00 | false-verified 0 |
| hadith exact / plain | 60 each | 0.92 | 1.00 | 0.88 | |
| hadith unmarked | 60 | 0.73 | 1.00 | 0.70 | |
| hadith spliced / fabricated | 60 each | 0.97 / 1.00 | 0.97 / 1.00 | 1.00 | false-verified 0 |

No fabricated or spliced text was ever marked verified in this run.

## Performance (measured)

Load time was traced to **duplicate downloads** (preload/prefetch plus the worker's own fetch; Hugging Face ignores `_headers`) and a large Hadith index,
not to computation. Fixes: no duplicate fetches, Hadith index fetched in parallel with the Python runtime and loaded lazily, per-build Cache API,
smaller indexes (`hadith.idx.gz` 13.6 MB → 8.2 MB, `quran.idx.gz` 2.1 MB).

Pyodide 0.26.4 executed under Node 22 with the final build files (CPU time only; runtime read from local disk), two runs:

| Step | Run 1 | Run 2 |
|---|---|---|
| Pyodide runtime ready | 2.2 s | 1.6 s |
| Quran pipeline ready | 2.8 s | 2.1 s |
| Hadith index loaded | 3.4 s | 2.6 s |
| 3 example verifications finished | 5.0 s | 4.5 s |

Download payload on a first visit: Pyodide core ≈ 13.6 MB (`pyodide.asm.wasm` 10.1 MB + stdlib 2.3 MB + loader) + Amanah files ≈ 10.3 MB; repeat visits use the browser cache.
Network time could not be measured from the authoring environment (Hugging Face, jsDelivr, PyPI were unreachable), so no end-to-end browser figure is claimed; the
page shows a small spinner while data loads. The ~15-minute load reported earlier is not reproduced by any computation measured here.

## Security

* No secret is stored in the repository, the page or the browser. The page calls a proxy; the free default uses Cloudflare Workers AI (no key). An OpenAI-compatible key, if used, is only a platform secret (`OPENAI_API_KEY`).
* The proxy limits prompt size (1,500 characters), answer length and origins (`*.hf.space`, `*.pages.dev`, `localhost`, or `ALLOWED_ORIGINS`), and keeps a per-IP limit. Add a Cloudflare rate-limiting rule.
* A test fails if a key-like pattern appears in the sources. If a key was ever pasted anywhere public, revoke it.

## Mapping to the judging criteria

| Criterion | Where it is addressed |
|---|---|
| Problem clarity and relevance (25%) | *The problem*; hallucinated religious quotations; three-way decision with human review |
| AI relevance and value (15%) | LLM answers are verified by a retrieval + alignment engine; optional CAMeLBERT merge; grounding guard |
| Documentation, reliability, scientific soundness (20%) | This file, `docs/`, measured metrics with stated limits, 69 tests, stress test, REFERENCES |
| Feasibility (15%) | Static site, free hosting, Pyodide in the browser, free Workers AI proxy |
| Originality (15%) | Abstention-first design, verbatim source corrections, corpus-first typing, grounding guard |
| Execution and task coverage (10%) | Subtasks 1A/1B/1C implemented and evaluated; working demo; Colab notebook |

## Limitations

* Unannounced-quotation detection is corpus matching, not understanding; very short or far-from-corpus quotations may be missed (Hadith without marks: 73% found in the stress test).
* Hadith are never auto-corrected; the system checks text against the bundled records, **not authenticity or grading**.
* A fine-tuned CAMeLBERT model must be trained (`research/train_detector.py`) and published by the team; none is bundled.
* The in-browser (Pyodide) build could not be run end to end in the authoring environment; its Python core was run under Node and its UI in a browser with a stubbed worker.
* The Hadith compilation's own licence is not stated upstream (the data comes from the IslamicEval repository, Apache-2.0, which cites a GPL-3.0 repository as its source); the Quran text must be kept unchanged (Tanzil, CC BY 3.0). See *Data sources* below and [docs/DATA_SOURCES_AND_LICENSES.md](docs/DATA_SOURCES_AND_LICENSES.md).
* A verification aid, not a religious ruling.

## Data sources

Verified against the upstream sources on 6 October 2026 (details and method: [docs/DATA_SOURCES_AND_LICENSES.md](docs/DATA_SOURCES_AND_LICENSES.md), notices: [NOTICE.md](NOTICE.md)).

| Data | Source | Terms |
|---|---|---|
| Quran text, `data/quran.json` | [Tanzil Project](https://tanzil.net) via the [IslamicEval 2025 Subtask 1 repository](https://github.com/qcri/IslamicEval-2025-Subtask-1) | CC BY 3.0 — verbatim only, credit and link to tanzil.net |
| Hadith, six books, `data/hadith.json` | Same repository; identical to `BookID` 1–6 of `nine_books_data.csv` in [OmarShafie/hadith](https://github.com/OmarShafie/hadith) | Repository Apache-2.0 / source repository GPL-3.0; no separate licence stated for the texts |
| 150 development responses | IslamicEval 2025 `dev_SubtaskA/B/C` | Apache-2.0; cite Mubarak et al. (2025) |

Code: MIT. Please cite: Mubarak et al. (2025), IslamicEval 2025; Tanzil Project; Shafie (2021), KASHAF.
