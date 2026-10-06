// Cloudflare Pages Function: /api/ask  (same origin as the site). Logic and secrets handling: server/ask-core.js
import { handleAsk } from "../../server/ask-core.js";

export const onRequest = ({ request, env }) => handleAsk(request, env);
