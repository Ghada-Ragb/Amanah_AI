from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

SYSTEM_PROMPT = (
    "أنت مساعد معرفي في العلوم الإسلامية. أجب بالعربية بإيجاز ودقة. عند الاستشهاد بآية قرآنية أو حديث نبوي اكتب نصه كاملًا "
    "بين علامتي تنصيص مزدوجتين \"...\" بعد عبارة تمهيدية مثل: قال الله تعالى: أو قال رسول الله ﷺ:. "
    "لا تضع بين علامات التنصيص إلا نص الآية أو الحديث، واذكر السورة ورقم الآية أو مصدر الحديث بعد الاقتباس."
    " اكتب بالحروف العربية فقط ولا تستخدم أي لغة أو كتابة أخرى (لا صينية ولا يابانية ولا إنجليزية). لا تذكر أكثر من حكم أو دليل لا تتأكد منه."
)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
MAX_PROMPT_CHARS = 1500

_FOREIGN = re.compile("[\u0400-\u04ff\u0900-\u097f\u0e00-\u0e7f\u1100-\u11ff\u3000-\u303f\u3040-\u30ff\u3130-\u318f"
                      "\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]+")


def sanitize_answer(text: str) -> str:
    text = _FOREIGN.sub(" ", text or "")
    text = re.sub(r"([،,؛.])\s*(?:[،,؛])+", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([،؛.:])", r"\1", text)
    return text.strip()


class LLMError(RuntimeError):
    """Raised with a user-presentable Arabic message."""


@dataclass
class LLMSettings:
    api_key: str = ""
    model: str = ""
    base_url: str = ""          
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(api_key=os.environ.get("OPENAI_API_KEY", ""), model=os.environ.get("OPENAI_MODEL", ""),
                   base_url=os.environ.get("OPENAI_BASE_URL", ""))


def build_request(settings: LLMSettings, prompt: str) -> Tuple[str, Dict[str, str], dict]:
    base = (settings.base_url or DEFAULT_BASE_URL).rstrip("/")
    body = {"model": settings.model.strip() or DEFAULT_MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {settings.api_key}"}
    return f"{base}/chat/completions", headers, body


def parse_response(payload: dict) -> str:
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("وصلت استجابة غير متوقعة من النموذج.") from exc
    if not text or not text.strip():
        raise LLMError("لم يُرجع النموذج أي نص.")
    return sanitize_answer(text)


_HTTP_MESSAGES = {
    400: "رفض المزوّد الطلب.",
    401: "خدمة الإجابة غير مهيّأة بعد (المفتاح غير صالح).",
    403: "الخدمة غير مصرّح لها باستخدام هذا النموذج.",
    404: "النموذج غير متاح لدى المزوّد.",
    429: "تجاوزت حد الاستخدام المسموح؛ حاول لاحقًا.",
}


def generate(settings: Optional[LLMSettings], prompt: str) -> str:
    settings = settings or LLMSettings.from_env()
    if not prompt or not prompt.strip():
        raise LLMError("اكتب سؤالًا أولًا.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise LLMError(f"السؤال طويل جدًا (الحد الأقصى {MAX_PROMPT_CHARS} حرف).")
    if not settings.api_key or not settings.api_key.strip():
        raise LLMError("خدمة الإجابة غير مهيّأة: لم يُضبط مفتاح الخادم.")
    url, headers, body = build_request(settings, prompt.strip())
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LLMError(_HTTP_MESSAGES.get(exc.code, f"فشل الطلب (الرمز {exc.code}).")) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMError("تعذّر الاتصال بالمزوّد؛ تحقق من الإنترنت.") from exc
    except json.JSONDecodeError as exc:
        raise LLMError("وصلت استجابة غير مقروءة من المزوّد.") from exc
    return parse_response(payload)
