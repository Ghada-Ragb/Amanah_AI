# Deployment

Arabic version: [DEPLOYMENT_AR.md](DEPLOYMENT_AR.md)

Amanah AI is a **static site** (`static_space/`, generated) plus an optional **question-answering proxy** for the Ask & Verify page.
Verification itself runs in the visitor's browser (Python via Pyodide), so it needs no server. The proxy logic is in `server/ask-core.js`; it runs as a free
**Cloudflare Worker** (`worker/`, for a Hugging Face Static Space) or as a **Cloudflare Pages Function** (`functions/api/ask.js`, same origin as the site).
No key is ever placed in the repository or the page.

## 1. Build the site

```bash
python -m unittest discover -s tests           
ICV_ASK_ENDPOINT=https\://icv-ask-proxy.ghada-islamic-verifier-2026.workers.dev python build_static_space.py
```

Output: `static_space/` (`index.html`, `config.json`, Python modules, `index/quran.idx.gz` 2.1 MB, `index/hadith.idx.gz` 8.2 MB, `demo/examples.json`, Space cards).
`askEndpoint` can also be edited afterwards in `static_space/config.json` without rebuilding. The indexes are committed (`index/`); rebuild them only after changing `data/`:
`python index_builder.py`.

## 2. The proxy — free, no key (Cloudflare Workers AI)

`worker/wrangler.toml` ships with `[ai] binding = "AI"` and `PROVIDER = "workers-ai"`. The Worker runs a model on Cloudflare's free tier. It tries `AI_MODEL` (if set) and then a built-in
list in order (starting with `@cf/meta/llama-3.3-70b-instruct-fp8-fast`), because Cloudflare retires models from time to time; check the current catalogue at
<https://developers.cloudflare.com/workers-ai/models/>. Free daily limits change too.

```bash
cd worker
npx wrangler login
npx wrangler deploy            # prints https://icv-ask-proxy.<account>.workers.dev
curl -s -X POST https://icv-ask-proxy.<account>.workers.dev -H "Content-Type: application/json" -d '{"prompt":"ما فضل الصبر؟"}'
```

`{"answer": ...}` = working. `{"error":"rate_limited"}` (429) = free quota or per-IP limit; `{"error":"upstream_ai"}` (502) = no listed model answered (to see the cause, temporarily uncomment
`DEBUG = "1"` in `wrangler.toml`, redeploy, and **remove it again**).

Alternative: any OpenAI-compatible API (OpenAI, Groq, Gemini compatibility endpoint): remove the `PROVIDER` line, set `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_TOKEN_PARAM`
(see comments in `wrangler.toml`) and store the key with `npx wrangler secret put OPENAI_API_KEY`. `{"error":"insufficient_quota"}` (402) = the provider account has no credit.

By default the proxy accepts browsers from `*.hf.space`, `*.pages.dev` and `localhost`. To restrict it to your Space set `ALLOWED_ORIGINS = "https://<user>-<space>.static.hf.space"` and redeploy. Add a Cloudflare rate-limiting rule.

## 3. Option A — Hugging Face Static Space

1. Create a Space of type **Static**.
2. Upload the **contents** of the Hugging Face ZIP (equivalently of `static_space/`): `index.html`, `config.json`, `README.md` (carries the Space front matter), the `.py` files, `index/`, `demo/`, `_headers`.
3. `index/*.idx.gz` files are larger than 10 MB in total; if the upload is refused for size, use Git LFS (`git lfs install && git lfs track "*.gz"`) before the first commit.
4. Open the Space URL. First visit downloads Pyodide and the indexes (a small spinner is shown under the input box); later visits use the browser cache.

## 4. Option B — Cloudflare Pages (site and function together)

1. Push the repository to GitHub (public). `.env`, `.dev.vars` and `static_space/` are git-ignored.
2. Cloudflare → Workers & Pages → Create → Pages → connect the repository. Build command `python build_static_space.py`, output directory `static_space`, root directory empty (so `/functions` is picked up).
3. Optional: set secrets (`OPENAI_API_KEY`, `ALLOWED_ORIGINS`) under Settings → Variables and Secrets; with the `AI` binding configured in Pages the Workers AI path is used. Otherwise point `askEndpoint` at a deployed Worker.
4. Test `POST https://<project>.pages.dev/api/ask` as in section 2.

## 5. Pre-demo checklist

* The small loading spinner disappears and *Verify directly* works with the built-in examples; *جرّب مثالًا* works on both pages.
* *Ask & Verify* returns a live answer. If the page says it could not obtain an answer, the proxy is unreachable or not configured (503 `not_configured`); the saved example button still demonstrates the verification flow.
* The copy button copies the corrected text and shows «تم نسخ النص بنجاح».
* `DEBUG` is **not** set in `wrangler.toml`.

## 6. Optional: hosted CAMeLBERT-MSA detector

1. Fine-tune and publish a token-classification model: `python research/train_detector.py --model CAMeL-Lab/bert-base-arabic-camelbert-msa` (labels `O, B-Ayah, I-Ayah, B-Hadith, I-Hadith`), then upload it to the Hugging Face Hub.
2. Edit `static_space/config.json`: `"hf": {"model": "YOUR-ORG/your-model", "endpoint": "https://router.huggingface.co/hf-inference/models/{model}"}`. The browser calls it without a token, so the model must be public; check the URL pattern in Hugging Face's current documentation.
3. The model's spans are merged silently with the bundled detector; without it (or if it fails) the bundled detector works alone.

## 7. Put the link in the documentation

Replace `[https://ghada-99-ragab-amanah-ai.static.hf.space]` in `README.md` and `README_AR.md` (search for `LIVE_DEPLOYMENT_URL`).
