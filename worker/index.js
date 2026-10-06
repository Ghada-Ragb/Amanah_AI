// Standalone Cloudflare Worker for static hosts without a server (for example a free Hugging Face Static Space).
// Deploy:  cd worker && npx wrangler secret put OPENAI_API_KEY && npx wrangler deploy
// Then set "askEndpoint" in the site's config.json to the Worker URL (https://<name>.<account>.workers.dev).
import { handleAsk } from "../server/ask-core.js";

export default { fetch: (request, env) => handleAsk(request, env) };
