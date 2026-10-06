from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import llm_client
import ui

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "static_space"
PYODIDE_VERSION = "0.26.4"
PYTHON_FILES = ["app.py", "verifier.py", "retrieval.py", "index_builder.py", "normalization.py", "alignment.py",
                "similarity.py", "detector.py", "scanner.py", "idgham.py", "ui.py", "llm_client.py",
                "benchmark_format.py", "camelbert_adapter.py"]
DATA_FILES = ["index/quran.idx.gz", "demo/examples.json"]
LAZY_FILES = ["index/hadith.idx.gz"]

DEFAULT_CONFIG = {

    "askEndpoint": os.environ.get("ICV_ASK_ENDPOINT", "/api/ask"),
    "hf": {"model": "", "endpoint": "https://router.huggingface.co/hf-inference/models/{model}"},
}

SPACE_README_EN = f"""---
title: Amanah AI
emoji: 🕌
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: AI-Powered Verification and Correction of Quranic and Prophetic Quotations
---

# Amanah AI

**{ui.ENGLISH_TITLE}**

Finds Quran verses and Hadith in any text (for example a language-model answer), compares them word by word with the
bundled sources, corrects Quranic wording verbatim from the source text, and refers anything uncertain to human review.
It never invents a verse, hadith, reference or correction: with insufficient evidence it abstains.

* **Ask & Verify** (default page): ask a question, get an answer, and see every quotation in it checked.
* **Verify directly**: paste any text.
* Every page has a **«جرّب مثالًا»** button with saved scenarios (all correct, all wrong, mixed, uncertain).

The whole verification pipeline runs inside your browser (Python via Pyodide); pasted text is not sent to any server.
Only the optional question-answering step calls a small server-side proxy. Arabic version: [README_AR.md](README_AR.md).

**Data:** Quran text from the [Tanzil Project](https://tanzil.net) (CC BY 3.0, unchanged); Hadith (six books) and development data as distributed in the
[IslamicEval 2025 Subtask 1 repository](https://github.com/qcri/IslamicEval-2025-Subtask-1) (Apache-2.0), the Hadith texts originating from
[OmarShafie/hadith](https://github.com/OmarShafie/hadith) (cite Shafie, 2021). Code: MIT.
"""

SPACE_README_AR = f"""# {ui.APP_TITLE}

**{ui.APP_TAGLINE}**

يكتشف الآيات والأحاديث في أي نص، ويطابقها كلمةً بكلمة مع المصادر المضمّنة، ويصحّح الآيات من نص المصدر نفسه،
ويحيل كل ما فيه شك إلى المراجعة البشرية. لا يخترع آية ولا حديثًا ولا مرجعًا ولا تصحيحًا؛ وعند نقص الدليل يمتنع عن الجزم.

* **اسأل ثم تحقّق** (الصفحة الافتراضية) · **تحقّق مباشر** · زر **«جرّب مثالًا»** في الصفحتين.

يعمل التحقق كاملًا داخل متصفحك (بايثون عبر Pyodide) ولا يُرسل النص الملصوق إلى أي خادم.
النسخة الإنجليزية: [README.md](README.md).

**البيانات:** نص القرآن من [مشروع Tanzil](https://tanzil.net) (CC BY 3.0، دون تغيير)؛ والحديث (الكتب الستة) وبيانات التطوير كما وُزّعت في
[مستودع IslamicEval 2025](https://github.com/qcri/IslamicEval-2025-Subtask-1) (Apache-2.0)، ونصوص الحديث أصلها [OmarShafie/hadith](https://github.com/OmarShafie/hadith). الكود: MIT.
"""

HEADERS = """/*
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer
/index/*
  Cache-Control: public, max-age=86400
/config.json
  Cache-Control: no-store
"""

PAGE_CSS = """
body.icv-page{margin:0;min-height:100vh;background-attachment:fixed;} .page{max-width:980px;margin:0 auto;padding:10px 16px 28px;box-sizing:border-box;}
.glass{background:var(--glass);border:1px solid var(--glass-line);border-radius:20px;box-shadow:var(--shadow-1);
  backdrop-filter:blur(16px) saturate(150%);-webkit-backdrop-filter:blur(16px) saturate(150%);}
.tabs{display:flex;gap:6px;margin:6px auto 14px;direction:rtl;padding:6px;border-radius:18px;max-width:560px;}
.tab{flex:1;border:0;background:transparent;color:var(--green);border-radius:13px;padding:11px 14px;font:700 1rem 'Cairo',sans-serif;cursor:pointer;transition:background .2s,color .2s,box-shadow .2s;}
.tab:hover{background:rgba(15,76,58,.07);}
.tab.active{background:linear-gradient(135deg,var(--green),var(--green-2));color:#FFF;box-shadow:0 6px 16px rgba(15,76,58,.28);}
.tab:focus-visible,.btns button:focus-visible{outline:3px solid rgba(184,145,47,.55);outline-offset:2px;}
section[hidden]{display:none;}
.form{padding:18px 20px 14px;margin-bottom:14px;} .form .icv{margin-bottom:6px;}
.form label{display:block;color:var(--muted);font-size:.88rem;font-weight:600;margin:10px 0 5px;direction:rtl;text-align:right;}
.form textarea,.form select{width:100%;box-sizing:border-box;background:rgba(255,255,255,.88);color:var(--ink);border:1px solid var(--line);border-radius:14px;padding:12px 14px;font:1rem 'Cairo',sans-serif;direction:rtl;text-align:right;outline:none;transition:border-color .2s,box-shadow .2s;}
.form textarea{min-height:200px;resize:vertical;font-family:'Amiri','Cairo',serif;font-size:1.25rem;line-height:2.1;}
.form textarea.short{min-height:96px;}
.form textarea:focus,.form select:focus{border-color:var(--gold);box-shadow:0 0 0 4px rgba(184,145,47,.18);}
.hint{font-size:.84rem;color:var(--muted);direction:rtl;margin:2px 0 0;}
.btns{display:flex;gap:12px;margin:14px 0 6px;flex-wrap:wrap;justify-content:center;}
.btns button{flex:1 1 200px;border:0;border-radius:14px;padding:13px 18px;font:700 1.02rem 'Cairo',sans-serif;cursor:pointer;transition:transform .15s,box-shadow .15s,filter .15s;}
.btn-main{background:linear-gradient(135deg,var(--green),var(--green-2));color:#FFF;box-shadow:0 8px 20px rgba(15,76,58,.28);}
.btn-main:hover{transform:translateY(-1px);filter:brightness(1.07);}
.btn-alt{background:rgba(255,255,255,.85);color:var(--green);border:1.5px solid var(--gold)!important;} .btn-alt:hover{background:#FFFBEF;transform:translateY(-1px);}
.btns button:disabled{opacity:.5;cursor:not-allowed;transform:none;}
#status{direction:rtl;text-align:center;color:var(--muted);font-size:.93rem;margin:8px 0;} #status:empty{display:none;}
.boot{display:flex;align-items:center;justify-content:center;gap:10px;margin:2px 0 12px;direction:rtl;color:var(--muted);font-size:.9rem;min-height:26px;transition:opacity .35s ease;}
.boot .dot{width:18px;height:18px;border-radius:50%;border:2.5px solid rgba(15,76,58,.16);border-top-color:var(--gold);animation:icv-spin .8s linear infinite;flex:none;}
.boot.ready{opacity:0;pointer-events:none;height:0;min-height:0;margin:0;overflow:hidden;}
@media (prefers-reduced-motion: reduce){.boot .dot{animation-duration:2s;}}
.banner{margin:0 0 12px;}
"""

WORKER_JS = """
// Python (Pyodide) and the whole verification pipeline run here, off the main thread.
let py = null, booted = null, hadithReady = null;
const post = (message) => self.postMessage(message);
const step = (name, state, extra) => post(Object.assign({ type: "step", name, state }, extra || {}));

async function openCache(buildId) {
  try {
    const names = await caches.keys();
    await Promise.all(names.filter((n) => n.startsWith("icv-") && n !== "icv-" + buildId).map((n) => caches.delete(n)));   // purge old builds
    return await caches.open("icv-" + buildId);
  } catch (e) { return null; }
}

async function fetchBytes(url, cache, onProgress) {
  if (cache) { try { const hit = await cache.match(url); if (hit) return { bytes: new Uint8Array(await hit.arrayBuffer()), cached: true }; } catch (e) { /* fall through */ } }
  const response = await fetch(url);
  if (!response.ok) throw new Error("تعذّر تحميل " + url + " (" + response.status + ")");
  if (cache) { try { await cache.put(url, response.clone()); } catch (e) { /* the cache is an optimisation only */ } }
  const total = Number(response.headers.get("Content-Length")) || 0;
  if (!response.body || !onProgress) return { bytes: new Uint8Array(await response.arrayBuffer()), cached: false };
  const reader = response.body.getReader(), chunks = []; let received = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); received += value.length; onProgress(received, total); }
  const bytes = new Uint8Array(received); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return { bytes, cached: false };
}

// Accept both layouts: files inside folders (index/, demo/) or uploaded flat next to index.html.
async function fetchAny(file, base, cache, onProgress) {
  let lastError = null;
  for (const path of [file, file.split("/").pop()]) {
    try { return await fetchBytes(new URL(path, base).href, cache, onProgress); } catch (error) { lastError = error; }
  }
  throw lastError;
}

// A server that sets Content-Encoding: gzip on .gz files makes the browser inflate them; Python expects gzip, so re-pack.
async function ensureGzip(file, bytes) {
  if (!file.endsWith(".gz") || (bytes[0] === 0x1f && bytes[1] === 0x8b)) return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function boot(message) {
  const cache = await openCache(message.buildId);
  step("python", "active");
  importScripts("https://cdn.jsdelivr.net/pyodide/v" + message.pyodideVersion + "/full/pyodide.js");
  const hadithFile = message.lazy[0];
  // the large Hadith index starts downloading now, in parallel with the Python runtime
  const hadithDownload = fetchAny(hadithFile, message.base, cache, (got, total) => step("hadith", "active", { pct: total ? Math.round((got / total) * 100) : null, mb: (got / 1048576).toFixed(1) }));
  hadithDownload.catch(() => {});
  py = await loadPyodide();
  step("python", "done");
  step("code", "active", { pct: 0 });
  for (const dir of ["/app", "/app/index", "/app/demo", "/app/data"]) py.FS.mkdir(dir);
  let cached = true, n = 0;
  for (const file of message.files) {
    const result = await fetchAny(file, message.base, cache);
    cached = cached && result.cached;
    py.FS.writeFile("/app/" + file, await ensureGzip(file, result.bytes));
    step("code", "active", { pct: Math.round((++n / message.files.length) * 100) });
  }
  const examplesText = new TextDecoder().decode(py.FS.readFile("/app/demo/examples.json"));
  py.runPython("import sys; sys.path.insert(0, '/app')");
  py.runPython("from app import get_pipeline, verify_text, verify_generated_answer; get_pipeline()");
  step("code", "done", { cached });
  post({ type: "ready", examples: JSON.parse(examplesText) });
  hadithReady = (async () => {
    const result = await hadithDownload;
    step("hadith", "active", { pct: 100, prepare: true });
    py.FS.writeFile("/app/" + hadithFile, await ensureGzip(hadithFile, result.bytes));
    py.runPython("get_pipeline().retriever.warm()");
    step("hadith", "done", { cached: result.cached });
    post({ type: "hadith" });
  })();
  await hadithReady;
}

self.onmessage = async (event) => {
  const message = event.data;
  if (message.type === "boot") {
    booted = boot(message);
    try { await booted; } catch (error) { post({ type: "error", text: String((error && error.message) || error) }); }
  } else if (message.type === "verify") {
    try {
      await booted;
      const html = py.globals.get(message.generated ? "verify_generated_answer" : "verify_text")(message.text, message.entities || "");
      post({ type: "result", id: message.id, html });
    } catch (error) { post({ type: "result", id: message.id, html: null }); }
  }
};
"""

PAGE_JS = """
const DEFAULT_CONFIG = __CONFIG__;
const BUILD_ID = "__BUILD_ID__";
const SKELETON = __SKELETON__;
const MAX_PROMPT = __MAX_PROMPT__;
const $ = (id) => document.getElementById(id);
let cfg = DEFAULT_CONFIG, examples = [], exampleIndex = 0, askExampleIndex = 0, nextId = 0, worker = null, ready = false;
const pending = new Map();

const store = {
  get(key) { try { return localStorage.getItem(key); } catch (e) { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* private mode / quota: caching is optional */ } },
};
function purgeOldLocalCache() {
  try { for (const key of Object.keys(localStorage)) if (key.startsWith("icv:examples:") && key !== "icv:examples:" + BUILD_ID) localStorage.removeItem(key); } catch (e) { /* ignore */ }
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function setStatus(text, busy = true) { $("status").innerHTML = (busy && text ? '<span class="icv-spinner"></span>' : "") + esc(text); }
function setBusy(flag) { for (const id of ["verify", "example", "ask", "ask-example"]) $(id).disabled = flag; }
function notice(text, kind) { return '<div class="icv banner"><div class="notice ' + kind + '">' + esc(text) + "</div></div>"; }
function show(html, banner) { $("results").innerHTML = (banner || "") + html; }
function showSkeleton() { $("results").innerHTML = SKELETON; }

function verifyRemote(payload) {
  return new Promise((resolve) => { const id = ++nextId; pending.set(id, resolve); worker.postMessage(Object.assign({ type: "verify", id }, payload)); });
}

// ---- detection: the bundled rule + corpus detector always runs; a fine-tuned CAMeLBERT-MSA hosted on Hugging Face (when configured)
// runs alongside it and both results are merged inside the pipeline. Nothing here is visible to the visitor and any failure is silent.
async function hostedEntities(text) {
  const model = (cfg.hf && cfg.hf.model || "").trim();
  if (!model) return "";
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 6000);
  try {
    const response = await fetch(cfg.hf.endpoint.replace("{model}", encodeURIComponent(model)), {
      method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
      body: JSON.stringify({ inputs: text, parameters: { aggregation_strategy: "simple" } }),
    });
    if (!response.ok) return "";
    const data = await response.json();
    return Array.isArray(data) ? JSON.stringify(data) : "";
  } catch (e) { return ""; } finally { clearTimeout(timer); }
}

async function verifyAuto(text, generated) {
  const entities = await hostedEntities(text);
  return verifyRemote({ text, generated, entities });
}

// Small models sometimes drift into Chinese etc.; strip scripts that never belong in an Arabic answer (the server does it too).
const FOREIGN = /[\\u0400-\\u04ff\\u0900-\\u097f\\u0e00-\\u0e7f\\u1100-\\u11ff\\u3000-\\u303f\\u3040-\\u30ff\\u3130-\\u318f\\u3400-\\u4dbf\\u4e00-\\u9fff\\uac00-\\ud7af\\uf900-\\ufaff\\uff00-\\uffef]+/g;
const cleanAnswer = (t) => String(t).replace(FOREIGN, " ").replace(/([،,؛.])\\s*[،,؛]+/g, "$1").replace(/[ \\t]{2,}/g, " ").replace(/ +([،؛.:])/g, "$1").trim();

async function runDirect() {
  if (!$("text").value.trim()) return show(notice("الرجاء إدخال نص للتحقق منه.", "warn"));
  setBusy(true); showSkeleton(); setStatus(ready ? "جارٍ التحقق…" : "جارٍ تجهيز النظام ثم التحقق…");
  const html = await verifyAuto($("text").value, false);
  show(html !== null ? html : notice("حدث خطأ غير متوقع أثناء التحقق.", "bad"));
  setStatus("", false); setBusy(false);
}

// ---- ask then verify: same-origin proxy that holds the key; saved sample answer when it is not reachable ----------------
async function askServer(prompt) {
  const response = await fetch(cfg.askEndpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt }) });
  if (!response.ok) {
    const error = new Error("http_" + response.status); error.status = response.status;
    try { error.code = (await response.json()).error; } catch (e) { error.code = ""; }
    throw error;
  }
  const data = await response.json();
  if (!data.answer) throw new Error("empty");
  return cleanAnswer(data.answer);
}

async function runAsk() {
  const prompt = $("prompt").value.trim();
  if (!prompt) return show(notice("اكتب سؤالًا أولًا.", "warn"));
  if (prompt.length > MAX_PROMPT) return show(notice("السؤال طويل جدًا (الحد الأقصى " + MAX_PROMPT + " حرف).", "warn"));
  setBusy(true); showSkeleton(); setStatus("جارٍ إعداد الإجابة…");
  let answer = null, banner = "";
  try { answer = await askServer(prompt); }
  catch (error) {
    const stop = (message) => { show(notice(message, "warn")); setStatus("", false); setBusy(false); };
    if (error.status === 402) return stop("رصيد خدمة الإجابة غير كافٍ حاليًا؛ على صاحب الحساب مراجعة الحصة المتاحة ثم المحاولة مجددًا.");
    if (error.status === 429) return stop("الطلبات كثيرة الآن؛ انتظر قليلًا ثم أعد المحاولة.");
    if (error.status === 413) return stop("السؤال طويل جدًا.");
    if (error.code === "upstream_auth") return stop("مفتاح الخدمة على الخادم غير صالح؛ يلزم تحديثه من صاحب الحساب.");
    return stop("تعذّر الوصول إلى خدمة الإجابة الآن؛ يمكنك تجربة أحد الأمثلة المحفوظة بزر «جرّب مثالًا» أو إعادة المحاولة بعد قليل.");
  }
  if (answer === null) { show(notice("تعذّر الحصول على إجابة الآن؛ حاول لاحقًا.", "warn")); setStatus("", false); return setBusy(false); }
  setStatus("جارٍ التحقق من الإجابة…");
  const html = await verifyAuto(answer, true);
  show(html !== null ? html : notice("تعذّر التحقق من الإجابة.", "bad"), banner);
  setStatus("", false); setBusy(false);
}

// Saved scenario: fills the question and verifies a stored model-style answer (no live call), so every case can be tried at once.
async function runAskExample() {
  const samples = examples.filter((e) => e.question);
  if (!samples.length) return show(notice("الأمثلة لم تُحمَّل بعد؛ انتظر لحظة ثم أعد المحاولة.", "warn"));
  const sample = samples[askExampleIndex++ % samples.length];
  $("prompt").value = sample.question;
  setBusy(true); showSkeleton(); setStatus("جارٍ التحقق من الإجابة…");
  const html = await verifyAuto(sample.text, true);
  show(html !== null ? html : notice("تعذّر التحقق من الإجابة.", "bad"),
    notice("إجابة تجريبية محفوظة لحالة «" + sample.title + "»، لم تُرسل إلى أي خدمة. لتجربة إجابة حيّة اكتب سؤالك واضغط «اسأل ثم تحقّق».", ""));
  setStatus("", false); setBusy(false);
}

// ---- boot panel ------------------------------------------------------------------------------------------------------
const STEP_LABELS = { python: "جارٍ تحميل البيانات…", code: "جارٍ تحميل البيانات…", hadith: "جارٍ تحميل كتب الحديث…" };
const stepsDone = new Set();
function applyStep(m) {
  if (m.state === "done") stepsDone.add(m.name);
  const text = $("boot-text");
  if (m.state === "active" && STEP_LABELS[m.name]) text.textContent = STEP_LABELS[m.name] + (m.name === "hadith" && m.pct != null && m.pct < 100 ? " " + m.pct + "%" : "");
  if (stepsDone.size >= 3) $("boot").classList.add("ready");
}

function start() {
  try { worker = new Worker(URL.createObjectURL(new Blob([$("worker-src").textContent], { type: "text/javascript" }))); }
  catch (error) { return setStatus("تعذّر تشغيل النظام: " + error.message, false); }
  worker.onerror = (event) => setStatus("تعذّر تشغيل النظام: " + (event.message || ""), false);
  worker.onmessage = (event) => {
    const m = event.data;
    if (m.type === "step") applyStep(m);
    else if (m.type === "error") setStatus("تعذّر تشغيل النظام: " + m.text + ". جرّب متصفح كمبيوتر حديثًا ثم أعد تحميل الصفحة.", false);
    else if (m.type === "result") { const resolve = pending.get(m.id); pending.delete(m.id); resolve(m.html); }
    else if (m.type === "ready") { examples = m.examples; ready = true; store.set("icv:examples:" + BUILD_ID, JSON.stringify(m.examples)); }
  };
  worker.postMessage({ type: "boot", base: location.href, buildId: BUILD_ID, pyodideVersion: "__PYODIDE_VERSION__", files: __FILES__, lazy: __LAZY__ });
}

async function loadConfig() {
  try {
    const response = await fetch("config.json", { cache: "no-store" });
    if (response.ok) { cfg = Object.assign({}, DEFAULT_CONFIG, await response.json()); store.set("icv:config", JSON.stringify(cfg)); return; }
  } catch (e) { /* offline or missing: fall back to the last known configuration */ }
  try { const saved = store.get("icv:config"); if (saved) cfg = Object.assign({}, DEFAULT_CONFIG, JSON.parse(saved)); } catch (e) { /* ignore */ }
}

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => { const on = t === tab; t.classList.toggle("active", on); t.setAttribute("aria-selected", on); });
  for (const name of ["direct", "ask"]) $("tab-" + name).hidden = tab.dataset.tab !== name;
}));
$("verify").addEventListener("click", runDirect);
$("ask").addEventListener("click", runAsk);
$("ask-example").addEventListener("click", runAskExample);
$("example").addEventListener("click", () => { if (!examples.length) return; $("text").value = examples[exampleIndex++ % examples.length].text; runDirect(); });

purgeOldLocalCache();
try { const saved = store.get("icv:examples:" + BUILD_ID); if (saved) examples = JSON.parse(saved); } catch (e) { examples = []; }
loadConfig();
start();
"""


def _digest() -> str:
    """Short content hash of everything the page loads: a new build gets a new cache name, so stale data never lingers."""
    h = hashlib.sha256()
    for name in sorted(PYTHON_FILES + DATA_FILES + LAZY_FILES):
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:12]


def build_index(build_id: str) -> str:
    js = (PAGE_JS.replace("__CONFIG__", json.dumps(DEFAULT_CONFIG))
          .replace("__BUILD_ID__", build_id)
          .replace("__SKELETON__", json.dumps(ui.SKELETON, ensure_ascii=False))
          .replace("__MAX_PROMPT__", str(llm_client.MAX_PROMPT_CHARS))
          .replace("__PYODIDE_VERSION__", PYODIDE_VERSION)
          .replace("__FILES__", json.dumps(PYTHON_FILES + DATA_FILES))
          .replace("__LAZY__", json.dumps(LAZY_FILES)))
    pyodide = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/pyodide.js"
    return f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{ui.APP_TITLE} · {ui.APP_TAGLINE}</title>
<meta name="description" content="{ui.APP_TAGLINE}">
<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<style>{ui.CSS}{PAGE_CSS}</style>
</head>
<body class="icv-page">
<div class="page">
{ui.HERO}
<div class="tabs glass" role="tablist"><button class="tab active" role="tab" aria-selected="true" data-tab="ask">اسأل ثم تحقّق</button><button class="tab" role="tab" aria-selected="false" data-tab="direct">تحقّق مباشر</button></div>
<section id="tab-ask" class="form glass">
  <div class="icv"><div class="notice">اكتب سؤالًا، وسيجيب عنه النظام مباشرةً، ثم يفحص كل آية وحديث في الإجابة ويعرض الأخطاء والتصحيحات.</div></div>
  <label for="prompt">سؤالك</label>
  <textarea id="prompt" class="short" maxlength="{llm_client.MAX_PROMPT_CHARS}" placeholder="{ui.PROMPT_PLACEHOLDER}"></textarea>
  <div class="btns"><button id="ask" class="btn-main">اسأل ثم تحقّق</button><button id="ask-example" class="btn-alt">جرّب مثالًا</button></div>
</section>
<section id="tab-direct" class="form glass" hidden>
  <label for="text">النص المراد التحقق منه</label>
  <textarea id="text" placeholder="{ui.PLACEHOLDER}"></textarea>
  <div class="btns"><button id="verify" class="btn-main">تحقّق من النص</button><button id="example" class="btn-alt">جرّب مثالًا</button></div>
</section>
<div class="boot" id="boot" aria-live="polite"><i class="dot"></i><span id="boot-text">جارٍ تحميل البيانات…</span></div>
<div id="status" role="status"></div>
<div id="results" class="results"></div>
{ui.DISCLAIMER}
</div>
<script>{ui.COPY_JS}</script>
<script type="text/plain" id="worker-src">{WORKER_JS}</script>
<script>{js}</script>
</body>
</html>
"""


def main() -> None:
    for name in PYTHON_FILES + DATA_FILES + LAZY_FILES:
        if not (ROOT / name).is_file():
            raise SystemExit(f"Missing {name}; run `python index_builder.py` first" if name.startswith("index/") else f"Missing {name}")
    if OUT.exists():
        shutil.rmtree(OUT)
    for name in PYTHON_FILES + DATA_FILES + LAZY_FILES:
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, target)
    build_id = _digest()
    (OUT / "index.html").write_text(build_index(build_id), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    (OUT / "_headers").write_text(HEADERS, encoding="utf-8")
    (OUT / "README.md").write_text(SPACE_README_EN, encoding="utf-8")
    (OUT / "README_AR.md").write_text(SPACE_README_AR, encoding="utf-8")
    print(f"Static site written to {OUT} (build {build_id})")


if __name__ == "__main__":
    main()
