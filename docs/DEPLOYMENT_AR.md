# النشر

النسخة الإنجليزية: [DEPLOYMENT.md](DEPLOYMENT.md)

Amanah AI **موقع ثابت** (`static_space/`، يُولَّد) مع **وسيط اختياري للإجابة عن الأسئلة** لصفحة «اسأل ثم تحقّق». أما التحقق نفسه فيعمل في متصفح الزائر (بايثون عبر Pyodide) فلا يحتاج خادمًا.
منطق الوسيط في `server/ask-core.js`، ويعمل كـ **Cloudflare Worker** مجاني (`worker/`، لاستضافة Hugging Face Static Space) أو كـ **دالة Cloudflare Pages** (`functions/api/ask.js`، على نطاق الموقع نفسه).
لا يوضع أي مفتاح في المستودع ولا في الصفحة.

## 1. بناء الموقع

```bash
python -m unittest discover -s tests          
ICV_ASK_ENDPOINT=https\://icv-ask-proxy.ghada-islamic-verifier-2026.workers.dev python build_static_space.py
```

المخرجات: `static_space/` (`index.html` و`config.json` وملفات بايثون و`index/quran.idx.gz` ‏2.1 م.ب و`index/hadith.idx.gz` ‏8.2 م.ب و`demo/examples.json` وبطاقات الـSpace).
يمكن تعديل `askEndpoint` لاحقًا في `static_space/config.json` دون إعادة بناء. الفهارس مُودَعة في `index/`؛ أعد بناءها فقط بعد تغيير `data/`: `python index_builder.py`.

## 2. الوسيط — مجاني وبلا مفتاح (Cloudflare Workers AI)

يأتي `worker/wrangler.toml` مضبوطًا على `[ai] binding = "AI"` و`PROVIDER = "workers-ai"`. يشغّل الـWorker نموذجًا من الطبقة المجانية في Cloudflare. يجرّب `AI_MODEL` (إن وُجد) ثم قائمة مدمجة بالترتيب
(تبدأ بـ`@cf/meta/llama-3.3-70b-instruct-fp8-fast`) لأن Cloudflare تُحيل نماذج إلى التقاعد من حين لآخر؛ راجع الفهرس الحالي في <https://developers.cloudflare.com/workers-ai/models/>. وتتغيّر الحدود اليومية المجانية أيضًا.

```bash
cd worker
npx wrangler login
npx wrangler deploy            # يطبع https://icv-ask-proxy.<account>.workers.dev
curl -s -X POST https://icv-ask-proxy.<account>.workers.dev -H "Content-Type: application/json" -d '{"prompt":"ما فضل الصبر؟"}'
```

`{"answer": ...}` = يعمل. `{"error":"rate_limited"}` (429) = نفدت الحصة المجانية أو حدّ العنوان؛ `{"error":"upstream_ai"}` (502) = لم يُجب أي نموذج من القائمة (لمعرفة السبب فعّل مؤقتًا
`DEBUG = "1"` في `wrangler.toml` وأعد النشر ثم **احذفه ثانية**).

بديل: أي واجهة متوافقة مع OpenAI (OpenAI أو Groq أو نقطة التوافق في Gemini): احذف سطر `PROVIDER` واضبط `OPENAI_BASE_URL` و`OPENAI_MODEL` و`OPENAI_TOKEN_PARAM`
(انظر تعليقات `wrangler.toml`) وخزّن المفتاح بـ`npx wrangler secret put OPENAI_API_KEY`. و`{"error":"insufficient_quota"}` (402) = حساب المزوّد بلا رصيد.

يقبل الوسيط افتراضيًا المتصفحات على `*.hf.space` و`*.pages.dev` و`localhost`. لحصره في Space الخاص بك اضبط `ALLOWED_ORIGINS = "https://<user>-<space>.static.hf.space"` وأعد النشر. وأضف قاعدة تحديد معدّل في Cloudflare.

## 3. الخيار أ — Hugging Face Static Space

1. أنشئ Space من نوع **Static**.
2. ارفع **محتويات** ملف ZIP الخاص بـHugging Face (أي محتويات `static_space/`): `index.html` و`config.json` و`README.md` (فيه ترويسة الـSpace) وملفات `.py` و`index/` و`demo/` و`_headers`.
3. ملفات `index/*.idx.gz` مجتمعة أكبر من 10 م.ب؛ إن رُفض الرفع بسبب الحجم فاستخدم Git LFS (`git lfs install && git lfs track "*.gz"`) قبل أول إيداع.
4. افتح رابط الـSpace. أول زيارة تنزّل Pyodide والفهارس (تظهر دائرة تحميل صغيرة تحت صندوق الإدخال)؛ والزيارات التالية من ذاكرة المتصفح.

## 4. الخيار ب — Cloudflare Pages (الموقع والدالة معًا)

1. ادفع المستودع إلى GitHub (عام). الملفات `.env` و`.dev.vars` و`static_space/` مستثناة من git.
2. Cloudflare ← Workers & Pages ← Create ← Pages ← اربط المستودع. أمر البناء `python build_static_space.py`، ومجلد المخرجات `static_space`، ومجلد الجذر فارغ ليلتقط `/functions`.
3. اختياري: أضف الأسرار (`OPENAI_API_KEY` و`ALLOWED_ORIGINS`) من Settings ← Variables and Secrets؛ ومع ضبط ربط `AI` في Pages يُستعمل مسار Workers AI. وإلا فوجّه `askEndpoint` إلى Worker منشور.
4. اختبر `POST https://<project>.pages.dev/api/ask` كما في القسم 2.

## 5. فحوصات قبل العرض

* تختفي دائرة التحميل الصغيرة ويعمل «تحقّق مباشر» على الأمثلة المدمجة، ويعمل «جرّب مثالًا» في الصفحتين.
* «اسأل ثم تحقّق» يعيد إجابة حيّة. وإن ذكرت الصفحة تعذّر الحصول على إجابة فالوسيط غير متاح أو غير مهيّأ (503 `not_configured`)؛ ويبقى زر المثال المحفوظ يعرض مسار التحقق.
* زر النسخ يُنسخ النص المصحَّح ويظهر «تم نسخ النص بنجاح».
* `DEBUG` **غير** مضبوط في `wrangler.toml`.

## 6. اختياري: كاشف CAMeLBERT-MSA مستضاف

1. اضبط ونشر نموذج تصنيف رموز: `python research/train_detector.py --model CAMeL-Lab/bert-base-arabic-camelbert-msa` (الوسوم `O, B-Ayah, I-Ayah, B-Hadith, I-Hadith`) ثم ارفعه إلى Hugging Face Hub.
2. عدّل `static_space/config.json`: `"hf": {"model": "YOUR-ORG/your-model", "endpoint": "https://router.huggingface.co/hf-inference/models/{model}"}`. يستدعيه المتصفح دون رمز وصول فيجب أن يكون النموذج عامًّا؛ وتحقّق من نمط الرابط من توثيق Hugging Face الحالي.
3. تُدمج مقاطع النموذج بصمت مع الكاشف المدمج؛ وبدونه (أو عند تعطّله) يعمل الكاشف المدمج وحده.

## 7. ضع الرابط في الوثائق

استبدل `[https://ghada-99-ragab-amanah-ai.static.hf.space]` في `README.md` و`README_AR.md` (ابحث عن `LIVE_DEPLOYMENT_URL`).
