// Shared server-side logic for "Ask then verify": used by the Cloudflare Worker (worker/index.js, for a Hugging Face Static
// Space or any other static host) and by the Pages Function (functions/api/ask.js).
//
// Providers (chosen on the server, invisible to visitors):
//   * Cloudflare Workers AI (free tier, no key at all): bind `[ai] binding = "AI"` and set PROVIDER = "workers-ai".
//   * Any OpenAI-compatible API (OpenAI, Groq, Gemini's compatibility endpoint ...): OPENAI_API_KEY + OPENAI_BASE_URL + OPENAI_MODEL.
// The OpenAI-compatible key lives ONLY in the platform secret OPENAI_API_KEY. It is never in the repository, the page or the browser.
// The model is fixed on the server (OPENAI_MODEL), so the page has no model selector and exposes no model name.
const SYSTEM_PROMPT = "أنت مساعد معرفي في العلوم الإسلامية. أجب بالعربية بإيجاز ودقة. عند الاستشهاد بآية قرآنية أو حديث نبوي اكتب نصه كاملًا بين علامتي تنصيص مزدوجتين \"...\" بعد عبارة تمهيدية مثل: قال الله تعالى: أو قال رسول الله ﷺ:. لا تضع بين علامات التنصيص إلا نص الآية أو الحديث، واذكر السورة ورقم الآية أو مصدر الحديث بعد الاقتباس. اكتب بالحروف العربية فقط ولا تستخدم أي لغة أو كتابة أخرى (لا صينية ولا يابانية ولا إنجليزية). لا تذكر أكثر من حكم أو دليل لا تتأكد منه.";
// Small models sometimes drift into Chinese etc.; strip scripts that never belong in an Arabic answer.
const FOREIGN = /[\u0400-\u04ff\u0900-\u097f\u0e00-\u0e7f\u1100-\u11ff\u3000-\u303f\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]+/g;
export function sanitizeAnswer(text) {
  return String(text || "").replace(FOREIGN, " ").replace(/([،,؛.])\s*[،,؛]+/g, "$1").replace(/[ \t]{2,}/g, " ").replace(/ +([،؛.:])/g, "$1").trim();
}
const MAX_PROMPT_CHARS = 1500;
const DEFAULT_MODEL = "gpt-4o-mini";
const WINDOW_MS = 10 * 60 * 1000, MAX_PER_WINDOW = 30;   // best-effort per-IP limit (per isolate); add a Cloudflare rate-limiting rule too
const hits = new Map();

// Origins allowed to call the endpoint from a browser: same origin, Hugging Face Spaces, Pages, plus ALLOWED_ORIGINS (comma separated).
function originAllowed(origin, requestUrl, env) {
  if (!origin) return true;
  if (origin === new URL(requestUrl).origin) return true;
  const extra = String(env.ALLOWED_ORIGINS || "").split(",").map((s) => s.trim()).filter(Boolean);
  if (extra.includes(origin)) return true;
  if (extra.length) return false;   // an explicit list wins over the defaults
  try { const host = new URL(origin).hostname; return host === "localhost" || host.endsWith(".hf.space") || host.endsWith(".pages.dev"); }
  catch (e) { return false; }
}

function corsHeaders(origin) {
  return origin ? { "Access-Control-Allow-Origin": origin, "Vary": "Origin", "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Max-Age": "86400" } : {};
}

function reply(body, status, origin) {
  return new Response(JSON.stringify(body), { status, headers: Object.assign(
    { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" }, corsHeaders(origin)) });
}

function limited(request) {
  const ip = request.headers.get("CF-Connecting-IP") || "unknown", now = Date.now();
  const recent = (hits.get(ip) || []).filter((t) => now - t < WINDOW_MS);
  recent.push(now); hits.set(ip, recent);
  if (hits.size > 5000) hits.clear();
  return recent.length > MAX_PER_WINDOW;
}

// OpenAI reports an empty balance under several codes / messages; all of them mean "add credit", not "slow down".
const isQuota = (code, message) => /insufficient_quota|credit_balance_exhausted|billing_hard_limit/.test(code || "") || /no credits|exceeded your current quota|billing/i.test(message || "");

function upstreamUrl(env) {
  return (env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, "") + "/chat/completions";
}

async function callOpenAI(env, prompt) {
  const url = upstreamUrl(env);
  const init = { method: "POST", headers: { "Content-Type": "application/json", Authorization: "Bearer " + String(env.OPENAI_API_KEY).trim() },
    body: JSON.stringify(Object.assign({ model: env.OPENAI_MODEL || DEFAULT_MODEL,
      messages: [{ role: "system", content: SYSTEM_PROMPT }, { role: "user", content: prompt }] },
      // OPENAI_TOKEN_PARAM: "max_completion_tokens" (default, current OpenAI models) | "max_tokens" (older / some compatible APIs) | "none"
      (env.OPENAI_TOKEN_PARAM || "max_completion_tokens") === "none" ? {} : { [env.OPENAI_TOKEN_PARAM || "max_completion_tokens"]: 900 })) };
  let response = await fetch(url, init);
  if (response.status === 429) {   // a transient rate limit gets one retry; an exhausted balance (insufficient_quota) does not
    const detail = await response.clone().json().catch(() => ({}));
    if (!isQuota((detail.error || {}).code, (detail.error || {}).message)) { await new Promise((r) => setTimeout(r, 900)); response = await fetch(url, init); }
  }
  return response;
}

// Cloudflare retires models from time to time (a retired id answers with a "deprecated" / "not found" error), so the Worker tries
// the configured AI_MODEL first and then these candidates in order. Check the current catalog: https://developers.cloudflare.com/workers-ai/models/
const WORKERS_AI_MODELS = [
  "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
  "@cf/meta/llama-3.1-8b-instruct-fast",
  "@cf/meta/llama-3.2-3b-instruct",
  "@cf/zai-org/glm-4.7-flash",
  "@cf/openai/gpt-oss-20b",
];
const MODEL_GONE = /deprecat|not found|no such model|unknown model|does not exist|not available|5028|5007/i;

function extractAnswer(result) {
  if (typeof result === "string") return result;
  if (!result) return "";
  if (typeof result.response === "string") return result.response;
  const choice = result.choices && result.choices[0];
  if (choice && choice.message && choice.message.content) return choice.message.content;
  if (Array.isArray(result.output)) {   // Responses-style output (some models)
    const parts = [];
    for (const item of result.output) for (const part of (item.content || [])) if (part.text) parts.push(part.text);
    return parts.join("");
  }
  return "";
}

async function askWorkersAI(env, prompt, origin) {
  const candidates = [...new Set([env.AI_MODEL, ...WORKERS_AI_MODELS].filter(Boolean))];
  const tried = [];
  for (const model of candidates) {
    try {
      const result = await env.AI.run(model, {
        max_tokens: 900, messages: [{ role: "system", content: SYSTEM_PROMPT }, { role: "user", content: prompt }] });
      const answer = sanitizeAnswer(extractAnswer(result));
      if (!answer) { tried.push(model + ": empty answer"); continue; }
      return reply(env.DEBUG === "1" ? { answer, model } : { answer }, 200, origin);
    } catch (e) {
      const message = String((e && e.message) || e);
      if (/limit|quota|capacity|neurons/i.test(message) && !MODEL_GONE.test(message)) return reply({ error: "rate_limited" }, 429, origin);
      tried.push(model + ": " + message.slice(0, 120));
      if (!MODEL_GONE.test(message)) break;   // a real failure: do not hammer the other models
    }
  }
  return reply(env.DEBUG === "1" ? { error: "upstream_ai", tried } : { error: "upstream_ai" }, 502, origin);
}

export async function handleAsk(request, env) {
  const origin = request.headers.get("Origin");
  if (!originAllowed(origin, request.url, env)) return reply({ error: "origin_not_allowed" }, 403, null);
  if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: corsHeaders(origin) });
  if (request.method !== "POST") return reply({ error: "method_not_allowed" }, 405, origin);
  const useWorkersAI = Boolean(env.AI) && (env.PROVIDER === "workers-ai" || !env.OPENAI_API_KEY);
  if (!useWorkersAI && !env.OPENAI_API_KEY) return reply({ error: "not_configured" }, 503, origin);
  if (limited(request)) return reply(env.DEBUG === "1" ? { error: "rate_limited", source: "worker_ip_limit" } : { error: "rate_limited" }, 429, origin);

  let prompt = "";
  try { prompt = String((await request.json()).prompt || "").trim(); } catch (e) { return reply({ error: "bad_request" }, 400, origin); }
  if (!prompt) return reply({ error: "empty_prompt" }, 400, origin);
  if (prompt.length > MAX_PROMPT_CHARS) return reply({ error: "prompt_too_long" }, 413, origin);

  if (useWorkersAI) return askWorkersAI(env, prompt, origin);

  let upstream;
  try { upstream = await callOpenAI(env, prompt); } catch (e) { return reply({ error: "upstream_unreachable" }, 502, origin); }
  if (!upstream.ok) {
    const raw = await upstream.text();                       // read the body once, then parse it
    let detail = {}; try { detail = JSON.parse(raw); } catch (e) { /* not JSON */ }
    const info = detail.error || {}, code = info.code || "";
    if (upstream.status === 429) {
      const extra = env.DEBUG === "1" ? { source: "openai", code, detail: String(info.message || raw || "").slice(0, 300) } : {};
      return isQuota(code, info.message) ? reply(Object.assign({ error: "insufficient_quota" }, extra), 402, origin) : reply(Object.assign({ error: "rate_limited" }, extra), 429, origin);
    }
    if (upstream.status === 401) return reply({ error: "upstream_auth" }, 502, origin);   // wrong / revoked key on the server
    // OpenAI's own error text (never contains the key) helps to diagnose a wrong model name or an unsupported parameter
    const body = { error: "upstream_" + upstream.status, detail: String(info.message || raw || "").slice(0, 300) };
    if (env.DEBUG === "1") {   // temporary switch (wrangler.toml [vars] DEBUG = "1"); never leave it on in production
      const key = String(env.OPENAI_API_KEY || "");
      body.diag = { upstreamHost: new URL(upstreamUrl(env)).host, model: env.OPENAI_MODEL || DEFAULT_MODEL,
        contentType: upstream.headers.get("content-type"), server: upstream.headers.get("server"), requestId: upstream.headers.get("x-request-id"),
        keyLooksValid: /^sk-[A-Za-z0-9_-]{20,}$/.test(key), keyHasWhitespaceOrQuotes: /[\s"']/.test(key),
        keyLength: key.length, keyStartsWithSk: key.startsWith("sk-"), keyFirst3: key.slice(0, 3),
        // code points of characters a real key never contains (the key's valid characters are never reported)
        badCharCodes: [...new Set([...key].filter((c) => !/[A-Za-z0-9_-]/.test(c)).map((c) => c.codePointAt(0)))].slice(0, 8) };
    }
    return reply(body, 502, origin);
  }
  const data = await upstream.json();
  const answer = sanitizeAnswer((((data.choices || [])[0] || {}).message || {}).content);
  if (!answer) return reply({ error: "empty_answer" }, 502, origin);
  return reply({ answer }, 200, origin);
}
