from __future__ import annotations

import html
from typing import List, Optional


_e = html.escape

APP_TITLE = "Amanah AI"
APP_TAGLINE = "نظام ذكي للتحقق من الاقتباسات القرآنية والحديثية"
ENGLISH_TITLE = "AI-Powered Verification and Correction of Quranic and Prophetic Quotations"
COPIED_MESSAGE = "تم نسخ النص بنجاح"

GROUP_LABEL = {"verified": "موثّق", "mismatch": "غير مطابق", "review": "تحتاج مراجعة بشرية"}
TYPE_LABEL = {"Ayah": "آية قرآنية", "Hadith": "حديث نبوي"}
INDICATORS = [
    ("composite", "الدرجة المركبة"),
    ("coverage", "تغطية الكلمات"),
    ("lcs_ratio", "التسلسل النصي"),
    ("token_overlap", "تداخل الكلمات"),
    ("edit_sim", "تشابه الحروف"),
    ("diacritic_sim", "التشكيل"),
]
METHODS = {
    "substring_match": "الاقتباس واردٌ كاملًا في المصدر",
    "threshold_pass": "تجاوز عتبة التطابق",
    "threshold_fail": "دون عتبة التطابق",
    "borderline_multi_cov": "حالة حدّية: عدة مصادر متقاربة",
    "borderline_default": "حالة حدّية",
    "borderline_low_retrieval": "حالة حدّية: استرجاع ضعيف",
    "no_candidates": "لا توجد مصادر مرشحة",
    "empty_span": "نص فارغ",
    "error": "تعذّرت المعالجة",
}


def _pct(value: float) -> str:
    return f"{round(value * 100)}%"


def source_label(source: dict) -> str:
    if source["type"] == "Quran":
        start, end = source["ayah_start"], source["ayah_end"]
        verses = f"{start}" if start == end else f"{start}–{end}"
        return f"سورة {source['surah_name']} — الآية {verses}"
    return f"حديث رقم {source['hadithID']} — {source['title']}"


def reason_text(reason: dict) -> str:
    code = reason["code"]
    src = reason.get("source", "")
    if code == "exact_match":
        return f"تطابق تام مع {src} بعد تجاهل اختلافات الرسم والتشكيل."
    if code == "close_match":
        return f"تطابق شبه كامل مع {src}."
    if code == "altered_passage":
        n = reason.get("n", 0)
        how = "تغيّر ترتيب الكلمات" if reason.get("reordered") else f"{n} موضع مختلف"
        return f"النص يخالف {src} ({how}). التصحيح المقترح هو نص المصدر حرفيًا ولم يُولَّد."
    if code == "weak_match":
        return "وُجد مصدر قريب لكن التطابق غير كافٍ للحكم الآلي."
    if code == "ungrounded":
        return "تعذّر التثبّت من المرجع المقترح داخل المراجع المضمّنة، فأُسقط الاقتراح ولم يُعرض أي تصحيح؛ يلزم الرجوع إلى مختص."
    if code == "too_short":
        return "الاقتباس قصير جدًا (كلمتان فقط) فلا يمكن تحديد مصدره بيقين؛ يلزم الرجوع إلى مختص."
    if code == "no_source":
        return "لا يوجد في المراجع المضمّنة نصٌّ يشبه هذا الاقتباس، وقد يكون مختلَقًا."
    if code == "insufficient_evidence":
        return "الأدلة غير كافية: لا مصدر واضح، ودرجة اليقين منخفضة."
    if code == "candidate_not_strong":
        return f"وُجد مرشح ({src}، قوة المطابقة {round(reason.get('strength', 0) * 100)}%) لكن الدليل لا يكفي للتصحيح الآلي."
    if code == "hadith_altered":
        return (f"النص قريب جدًا من {src} لكن يختلف عنه في {reason.get('n', 1)} موضع (زيادة أو نقص أو استبدال كلمة)، وقد يغيّر ذلك المعنى؛ "
                "لا يُوثَّق الحديث إلا إذا طابق النص حرفيًا، ولا يُصحَّح آليًا فتلزم مراجعة مختص.")
    if code == "hadith_candidate":
        return f"وُجدت رواية مشابهة ({src}) لكن لا يُصحَّح الحديث آليًا لاختلاف الروايات؛ يلزم الرجوع إلى مختص."
    return "تعذّرت معالجة هذا الاقتباس آليًا."


def note_text(note: dict) -> str:
    n = note.get("n", 0)
    code = note["code"]
    if code == "diacritic_conflict":
        return f"تنبيه: تشكيل {n} كلمة يخالف المصحف (الكلمات صحيحة لكن الحركات مختلفة)."
    if code == "orthographic_variant":
        return f"ملاحظة: {n} اختلاف إملائي (رسم أو مسافات) لا يغيّر الكلمة."
    if note.get("misattributed"):
        if code == "is_ayah":
            return f"تنبيه: النص قرآني ({note['source']}) لكن عبارة التقديم تنسبه إلى الحديث."
        return f"تنبيه: النص حديث ({note['source']}) لكن عبارة التقديم تنسبه إلى القرآن."
    if code == "is_hadith":
        return f"تنبيه: هذا النص وارد في الحديث ({note['source']}) وليس في القرآن؛ ربما نُسب إلى الله تعالى خطأً."
    if code == "is_ayah":
        return f"تنبيه: هذا النص وارد في القرآن ({note['source']}) وليس في الحديث؛ ربما نُسب إلى النبي ﷺ خطأً."
    if code == "hadith_minor_diffs":
        return f"ملاحظة: {n} اختلاف طفيف عن أقرب رواية في المراجع."
    return ""


def _diff_html(comparison: dict) -> str:
    parts = []
    for op in comparison["word_diff"]:
        if op["op"] == "equal":
            parts.append(f'<span class="w-eq">{_e(op["span"])}</span>')
            continue
        if op["span"]:
            parts.append(f'<span class="w-extra">{_e(op["span"])}</span>')
        if op["source"]:
            parts.append(f'<span class="w-missing">{_e(op["source"])}</span>')
    return " ".join(parts)


def _legend(group: str, comparison: Optional[dict]) -> str:
    if group == "verified" or not comparison or all(op["op"] == "equal" for op in comparison["word_diff"]):
        return ""
    return '<p class="legend"><span class="w-extra">كلمات في الاقتباس تخالف المصدر</span> <span class="w-missing">الصواب من المصدر</span></p>'


def _indicator_table(signals: Optional[dict]) -> str:
    if not signals:
        return ""
    rows = []
    for key, label in INDICATORS:
        value = float(signals.get(key, 0.0))
        rows.append(
            f'<div class="ind"><div class="ind-name">{label}</div>'
            f'<div class="bar"><span style="width:{_pct(value)}"></span></div><div class="ind-val">{_pct(value)}</div></div>'
        )
    rows.append(f'<div class="ind"><div class="ind-name">وروده كاملًا في المصدر</div><div class="ind-val wide">{"نعم" if signals.get("is_substring") else "لا"}</div></div>')
    return '<div class="indicators">' + "".join(rows) + "</div>"


def _match_percent(span: dict) -> int:
    evidence = span["evidence"]
    if evidence and evidence.get("comparison"):
        return round(evidence["comparison"]["word_similarity"] * 100)
    if evidence and evidence.get("signals"):
        return round(evidence["signals"]["composite"] * 100)
    return 0


def _evidence_panel(span: dict) -> str:
    evidence, verification = span["evidence"], span["verification"]
    rows = [
        f'<div class="ev-row"><b>الاقتباس المكتشف</b><p class="quote">{_e(span["text"])}</p></div>',
        f'<div class="ev-row"><b>النوع</b><p>{TYPE_LABEL[span["type"]]}</p></div>',
    ]
    if evidence:
        comparison = evidence["comparison"]
        rows += [
            f'<div class="ev-row"><b>المصدر المرشح</b><p>{_e(source_label(evidence["source"]))}</p></div>',
            f'<div class="ev-row"><b>نص المصدر الأصلي</b><p class="quote">{_e(comparison["source_excerpt"])}</p></div>',
            f'<div class="ev-row"><b>المقارنة كلمةً بكلمة</b><p class="diff">{_diff_html(comparison)}</p>{_legend(span["group"], comparison)}</div>',
            f'<div class="ev-row"><b>مؤشرات التحقق</b>{_indicator_table(evidence.get("signals"))}</div>',
        ]
        if comparison["diacritic_notes"]:
            items = "، ".join(f"{_e(n['word'])} ← {_e(n['source_word'])}" for n in comparison["diacritic_notes"])
            rows.append(f'<div class="ev-row"><b>اختلافات التشكيل</b><p>{items}</p></div>')
    else:
        rows.append('<div class="ev-row"><b>المصدر المرشح</b><p>لم يُسترجع أي مصدر.</p></div>')
    rows += [
        f'<div class="ev-row"><b>درجة اليقين في الحكم</b><p>{_pct(verification["confidence"])} · {METHODS.get(verification["method"], "")}</p></div>',
        f'<div class="ev-row"><b>القرار</b><p>{_e(span["status_ar"])}</p></div>',
        f'<div class="ev-row"><b>سبب القرار</b><p>{_e(reason_text(span["reason"]))}</p></div>',
    ]
    return "".join(rows)


def _card(span: dict) -> str:
    group, evidence = span["group"], span["evidence"]
    label = GROUP_LABEL[group]
    chip = f'<span class="conf">نسبة المطابقة <b>{_match_percent(span)}%</b></span>'
    scan_badge = ('<span class="badge scan" title="اكتُشف بمطابقة النص مع المراجع دون علامات تنصيص">اقتباس غير معلَن</span>'
                  if span["detection"]["backend"] == "scan" else "")

    fields = [f'<div class="field"><label>الاقتباس المكتشف</label><p class="quote">{_e(span["text"])}</p></div>']
    if evidence:
        comparison = evidence["comparison"]
        fields.append(f'<div class="field"><label>المصدر المرشح</label><p>{_e(source_label(evidence["source"]))}</p></div>')
        if group != "verified":   
            fields.append(
                f'<div class="field"><label>المقارنة بالمصدر</label><p class="diff">{_diff_html(comparison)}</p>{_legend(group, comparison)}</div>'
            )
    else:
        fields.append('<div class="field"><label>المصدر المرشح</label><p>لا يوجد</p></div>')
    fields.append(f'<p class="reason">{_e(reason_text(span["reason"]))}</p>')

    notes = "".join(f'<p class="note">{_e(note_text(n))}</p>' for n in span["notes"] if note_text(n))
    action = ""
    correction, suggestion = span["correction"], span["suggestion"]
    if correction:
        action = (
            '<div class="action ok"><b>التصحيح المقترح من المصدر</b>'
            f'<p class="quote">{_e(correction["display_text"])}</p>'
            f'<p class="legend">{_e(source_label(correction["source"]))}</p></div>'
        )
    elif group == "review":
        closest = ""
        if suggestion:
            closest = (f'<p class="legend">أقرب مصدر وُجد للمراجِع، وليس تصحيحًا آليًا: {_e(source_label(suggestion["source"]))}</p>'
                       f'<p class="quote small">{_e(suggestion["display_text"][:420])}</p>')
        action = ('<div class="action warn"><b>الأدلة غير كافية للتصحيح الآلي.</b> يُوصى بالمراجعة البشرية.' + closest + "</div>")
    elif span["status"] == "UNSUPPORTED":
        action = '<div class="action bad"><b>لا يوجد مصدر مطابق في المراجع المتاحة.</b> لم يُقترح أي نص بديل.</div>'

    return f"""
<div class="qcard {group}">
  <div class="qhead">
    <span class="idx">{span["id"]}</span>
    <span class="badge type">{TYPE_LABEL[span["type"]]}</span>{scan_badge}
    <span class="badge st {group}">{label}</span>
    {chip}
  </div>
  {"".join(fields)}
  {notes}
  {action}
  <details class="evidence"><summary>عرض الدليل</summary><div class="ev-body">{_evidence_panel(span)}</div></details>
</div>"""


def _summary(summary: dict) -> str:
    tiles = [
        (summary["n_spans"], "إجمالي الاقتباسات", ""),
        (summary["n_ayah"], "آيات قرآنية", ""),
        (summary["n_hadith"], "أحاديث نبوية", ""),
        (summary["VERIFIED"], "موثّق", "verified"),
        (summary["CORRECTED"] + summary["UNSUPPORTED"], "غير مطابق", "mismatch"),
        (summary["HUMAN_REVIEW"], "تحتاج مراجعة", "review"),
    ]
    return '<div class="summary">' + "".join(
        f'<div class="tile {cls}{" zero" if value == 0 and cls else ""}"><b>{value}</b><span>{label}</span></div>'
        for value, label, cls in tiles
    ) + "</div>"


def _highlighted_text(result: dict, title: str) -> str:
    text, pieces, cursor = result["input_text"], [], 0
    for span in result["spans"]:
        pieces.append(_e(text[cursor:span["start"]]))
        pieces.append(f'<mark class="{span["group"]}">{_e(text[span["start"]:span["end"]])}</mark>')
        cursor = span["end"]
    pieces.append(_e(text[cursor:]))
    legend = ('<div class="hl-legend"><mark class="verified">موثّق</mark><mark class="mismatch">غير مطابق</mark>'
              '<mark class="review">مراجعة بشرية</mark></div>')
    return f'<div class="highlight"><label>{title}</label><p>' + "".join(pieces) + "</p>" + legend + "</div>"


def final_text(result: dict) -> str:
    text, reports = result["input_text"], result["spans"]
    for report in sorted(reports, key=lambda r: r["start"], reverse=True):
        if report["status"] == "CORRECTED" and report["correction"]:
            text = text[: report["start"]] + report["correction"]["display_text"] + text[report["end"]:]
        elif report["status"] == "UNSUPPORTED":
            text = text[: report["end"]] + " [⚠ لا يوجد مصدر مطابق]" + text[report["end"]:]
        elif report["status"] == "HUMAN_REVIEW":
            text = text[: report["end"]] + " [⚠ يحتاج مراجعة بشرية]" + text[report["end"]:]
    return text


def _final_block(result: dict) -> str:
    summary = result["summary"]
    fixed = summary["CORRECTED"]
    open_items = summary["UNSUPPORTED"] + summary["HUMAN_REVIEW"]
    if fixed == 0 and open_items == 0:
        headline = "كل الاقتباسات المكتشفة مطابقة للمصادر."
    else:
        headline = f"صُحِّح {fixed} اقتباس من نص المصدر، وبقي {open_items} اقتباس يحتاج مراجعة بشرية ومعلَّم بعلامة تنبيه."
    text = final_text(result)
    return (
        '<div class="final"><div class="final-head"><b>النسخة المصحّحة</b>'
        f'<button class="copy" type="button" data-text="{_e(text, quote=True)}" '
        'onclick="window.icvCopy&&window.icvCopy(this)">نسخ النص</button></div>'
        f'<p class="legend">{headline}</p><p class="final-text">{_e(text)}</p></div>'
    )


def render_results(result: dict, generated_answer: Optional[str] = None) -> str:
    header = render_generated_header(generated_answer) if generated_answer is not None else ""
    if not result["spans"]:
        body = (
            '<div class="notice">لم يُعثر على اقتباسات قرآنية أو حديثية في هذا النص. يتعرّف النظام على الاقتباسات بمطابقتها مع المراجع، '
            'سواء وُضعت بين علامات تنصيص أو أقواس أو وردت داخل الكلام دون أي عبارة تمهيدية.</div>'
        )
        return f'<div class="icv">{header}{body}</div>'
    generated = generated_answer is not None
    title = "إجابة النموذج مع الاقتباسات المكتشفة" if generated else "النص مع الاقتباسات المكتشفة"
    parts = [_summary(result["summary"]), _highlighted_text(result, title), "".join(_card(s) for s in result["spans"])]
    summary = result["summary"]
    if generated or summary["CORRECTED"] or summary["UNSUPPORTED"] or summary["HUMAN_REVIEW"]:
        parts.append(_final_block(result))
    return f'<div class="icv">{header}{"".join(parts)}</div>'


def render_generated_header(answer: str) -> str:
    return f'<div class="generated"><label>إجابة النموذج اللغوي كما وصلت</label><p>{_e(answer)}</p></div>'


def render_message(message: str, kind: str = "info") -> str:
    return f'<div class="icv"><div class="notice {kind}">{_e(message)}</div></div>'


STAR = (
    '<svg class="mark" viewBox="0 0 64 64" fill="none" stroke="#E2C06E" stroke-width="1.8" aria-hidden="true">'
    '<rect x="14" y="14" width="36" height="36"/><rect x="14" y="14" width="36" height="36" transform="rotate(45 32 32)"/>'
    '<circle cx="32" cy="32" r="7" fill="#E2C06E" stroke="none"/></svg>'
)

HERO = f"""
<div class="icv"><header class="hero">
  {STAR}
  <h1 class="brand">{APP_TITLE}</h1>
  <p class="tagline">{APP_TAGLINE}</p>
  <p class="sub">يكتشف الاقتباسات القرآنية والحديثية داخل أي نص، ويطابقها كلمةً بكلمة مع نصوص المراجع، ويصحّح الخطأ من المصدر نفسه.
     وحين لا يجد دليلًا كافيًا لا يخمّن أبدًا، بل يعلن أن الاقتباس غير مطابق أو يحيله إلى المراجعة البشرية.</p>
  <ul class="trust"><li><b>من المصدر</b> كل مرجع وتصحيح من نص المراجع حرفيًا</li><li><b>٣ قرارات</b> موثّق · غير مطابق · مراجعة بشرية</li><li><b>داخل المتصفح</b> التحقق لا يرسل نصك إلى أي خادم</li></ul>
</header></div>
"""

DISCLAIMER = """<div class="icv"><div class="disclaimer">أداة مساعدة للتدقيق النصي وليست فتوى ولا بديلًا عن المراجعة المتخصصة.
النتائج مبنية على مراجع القرآن الكريم والكتب الستة المضمّنة فقط.
<br>نص القرآن: <a href="https://tanzil.net" target="_blank" rel="noopener">مشروع Tanzil</a> (CC BY 3.0) · بيانات IslamicEval 2025.</div></div>"""

PLACEHOLDER = "الصق هنا النص الذي ولّده نموذج لغوي. يكتشف النظام الآيات والأحاديث الواردة فيه، بعلامات تنصيص أو بدونها…"
PROMPT_PLACEHOLDER = "اكتب سؤالك، مثل: اشرح لي فضل الصبر في القرآن والسنة مع ذكر الأدلة."

_PATTERN = (
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='88' height='88' viewBox='0 0 88 88'%3E"
    "%3Cg fill='none' stroke='%230F4C3A' stroke-opacity='0.07' stroke-width='1'%3E"
    "%3Crect x='22' y='22' width='44' height='44'/%3E%3Crect x='22' y='22' width='44' height='44' transform='rotate(45 44 44)'/%3E"
    "%3C/g%3E%3C/svg%3E\")"
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&family=Amiri:wght@400;700&display=swap');
:root { --green:#0F4C3A; --green-2:#17694F; --gold:#B8912F; --gold-2:#E2C06E; --cream:#FBF6EA; --paper:#FFFFFF;
        --ink:#1F2933; --muted:#55626D; --line:#E6DCC3; --ok:#1B7A4B; --ok-bg:#EAF6EF; --ok-line:#BFE3CD;
        --bad:#B3261E; --bad-bg:#FDECEA; --bad-line:#F4B8B3; --warn:#7A5B0C; --warn-bg:#FFF6DA; --warn-line:#EBD28A; }
.gradio-container, body.icv-page { background:var(--cream) PATTERN !important; font-family:'Cairo','Segoe UI',Tahoma,sans-serif; color:var(--ink); }
.gradio-container { max-width:1060px !important; --body-text-color:#1F2933; --block-background-fill:#FFFFFF; --block-border-color:#E6DCC3;
  --input-background-fill:#FFFFFF; --block-label-text-color:#55626D; --button-primary-background-fill:#0F4C3A;
  --button-primary-background-fill-hover:#17694F; --button-primary-text-color:#FFFFFF; --button-secondary-background-fill:#FFFFFF;
  --button-secondary-text-color:#0F4C3A; --button-secondary-border-color:#B8912F; --color-accent:#B8912F; }
.icv { direction:rtl; text-align:right; color:var(--ink); line-height:1.9; }
.icv .hero { background:linear-gradient(135deg,#0F4C3A,#17694F); border-radius:20px; padding:30px 22px 26px; margin:10px 0 18px; text-align:center;
  box-shadow:0 6px 22px rgba(15,76,58,.18); border-bottom:4px solid var(--gold); }
.icv .hero .mark { width:54px; height:54px; display:block; margin:0 auto 4px; }
.icv .hero h1 { font-size:2.3rem; margin:.1rem 0; color:#FFFFFF; font-weight:700; }
.icv .hero .tagline { color:var(--gold-2); font-size:1.2rem; font-weight:600; margin:.1rem 0 .6rem; }
.icv .hero .sub { max-width:720px; margin:0 auto; color:#E9F2EE; font-size:1rem; }
.icv .flow { display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap:8px; margin-top:16px; }
.icv .flow span { background:rgba(255,255,255,.12); border:1px solid rgba(226,192,110,.55); color:#FFF; border-radius:999px; padding:3px 16px; font-size:.9rem; }
.icv .flow i { color:var(--gold-2); font-style:normal; font-size:1.2rem; }
.icv .disclaimer { font-size:.85rem; color:var(--muted); text-align:center; padding:14px 8px; }
.icv .en-free { direction:rtl; }
.input-area textarea { direction:rtl; text-align:right; font-family:'Amiri','Cairo',serif !important; font-size:1.25rem !important; line-height:2.1 !important; background:#FFFFFF !important; color:#1F2933 !important; }
.llm-row label, .gradio-container label span { font-family:'Cairo',sans-serif; }
.results { direction:rtl; }
.icv .summary { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin:8px 0 14px; }
.icv .tile { background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:12px 8px; text-align:center; }
.icv .tile b { display:block; font-size:1.8rem; color:var(--green); line-height:1.3; } .icv .tile span { font-size:.92rem; color:var(--muted); }
.icv .tile.verified { background:var(--ok-bg); border-color:var(--ok-line); } .icv .tile.verified b { color:var(--ok); }
.icv .tile.mismatch { background:var(--bad-bg); border-color:var(--bad-line); } .icv .tile.mismatch b { color:var(--bad); }
.icv .tile.review { background:var(--warn-bg); border-color:var(--warn-line); } .icv .tile.review b { color:var(--warn); }
.icv .tile.zero { background:var(--paper); border-color:var(--line); opacity:.6; } .icv .tile.zero b { color:var(--muted); }
.icv .highlight, .icv .generated, .icv .final { background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:14px 18px; margin-bottom:14px; }
.icv label { display:block; color:var(--muted); font-size:.82rem; font-weight:600; margin-bottom:4px; }
.icv .highlight p, .icv .generated p, .icv .final-text { font-family:'Amiri','Cairo',serif; font-size:1.2rem; line-height:2.2; margin:0; white-space:pre-wrap; }
.icv mark { color:var(--ink); border-radius:6px; padding:1px 5px; }
.icv mark.verified { background:#D5EEDF; } .icv mark.mismatch { background:#F8CFCB; } .icv mark.review { background:#F7E3A6; }
.icv .hl-legend { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; font-size:.78rem; }
.icv .qcard { background:var(--paper); border:1px solid var(--line); border-inline-start:6px solid var(--line); border-radius:16px; padding:16px 18px; margin:12px 0; }
.icv .qcard.verified { border-inline-start-color:var(--ok); } .icv .qcard.mismatch { border-inline-start-color:var(--bad); } .icv .qcard.review { border-inline-start-color:var(--gold); }
.icv .qhead { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:8px; }
.icv .idx { background:var(--green); color:#FFF; border-radius:50%; width:28px; height:28px; display:inline-flex; align-items:center; justify-content:center; font-weight:700; font-size:.9rem; }
.icv .badge { padding:2px 14px; border-radius:999px; font-size:.84rem; background:#F3EEDD; border:1px solid var(--line); color:var(--ink); }
.icv .badge.st.verified { background:var(--ok-bg); border-color:var(--ok-line); color:var(--ok); font-weight:600; }
.icv .badge.st.mismatch { background:var(--bad-bg); border-color:var(--bad-line); color:var(--bad); font-weight:600; }
.icv .badge.st.review { background:var(--warn-bg); border-color:var(--warn-line); color:var(--warn); font-weight:600; }
.icv .badge.scan { background:#EEF3FB; border-color:#C9D8EE; color:#2F4F7F; }
.icv .conf { margin-inline-start:auto; color:var(--muted); font-size:.88rem; } .icv .conf b { color:var(--green); }
.icv .field { margin:8px 0; } .icv .field p { margin:0; }
.icv .quote { font-family:'Amiri','Cairo',serif; font-size:1.25rem; line-height:2.2; color:var(--ink); } .icv .quote.small { font-size:1.05rem; color:#3a4651; }
.icv .diff { font-family:'Amiri','Cairo',serif; font-size:1.2rem; line-height:2.2; }
.icv .w-extra { background:var(--bad-bg); color:var(--bad); border-radius:4px; padding:0 4px; text-decoration:line-through; }
.icv .w-missing { background:var(--ok-bg); color:var(--ok); border-radius:4px; padding:0 4px; font-weight:700; }
.icv .legend { color:var(--muted); font-size:.82rem; margin:4px 0 0; } .icv .legend span { text-decoration:none; font-size:.8rem; }
.icv .reason { color:var(--muted); font-size:.92rem; margin:6px 0 0; }
.icv .note { color:var(--warn); background:var(--warn-bg); border:1px solid var(--warn-line); border-radius:10px; padding:5px 12px; font-size:.88rem; margin:8px 0 0; }
.icv .action { border-radius:12px; padding:10px 14px; margin-top:10px; }
.icv .action.ok { background:var(--ok-bg); border:1px solid var(--ok-line); } .icv .action.ok b { color:var(--ok); }
.icv .action.warn { background:var(--warn-bg); border:1px solid var(--warn-line); } .icv .action.warn b { color:var(--warn); }
.icv .action.bad { background:var(--bad-bg); border:1px solid var(--bad-line); } .icv .action.bad b { color:var(--bad); }
.icv .evidence { margin-top:12px; border-top:1px dashed var(--line); padding-top:8px; }
.icv .evidence summary { cursor:pointer; color:var(--green); font-weight:700; }
.icv .ev-row { margin:10px 0; } .icv .ev-row b { color:var(--gold); font-size:.88rem; } .icv .ev-row p { margin:2px 0; }
.icv .indicators { display:grid; gap:6px; margin-top:6px; }
.icv .ind { display:grid; grid-template-columns:150px 1fr 48px; gap:10px; align-items:center; font-size:.88rem; }
.icv .bar { background:#EFE8D3; border-radius:999px; height:9px; overflow:hidden; direction:rtl; } .icv .bar span { display:block; height:100%; background:linear-gradient(270deg,var(--green),var(--gold)); }
.icv .ind-val { text-align:left; direction:ltr; color:var(--ink); } .icv .ind-val.wide { grid-column:2 / span 2; text-align:right; direction:rtl; }
.icv .final { border-color:var(--gold); background:#FFFDF6; }
.icv .final-head { display:flex; justify-content:space-between; align-items:center; } .icv .final-head b { color:var(--green); font-size:1.05rem; }
.icv .copy { background:var(--green); color:#FFF; border:0; border-radius:10px; padding:5px 16px; font-family:inherit; cursor:pointer; }
.icv .notice { background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:16px; }
.icv .notice.warn { background:var(--warn-bg); border-color:var(--warn-line); color:var(--warn); } .icv .notice.bad { background:var(--bad-bg); border-color:var(--bad-line); color:var(--bad); }
@media (max-width:640px){ .icv .hero h1{font-size:1.7rem;} .icv .ind{grid-template-columns:104px 1fr 40px;} }
""".replace("PATTERN", _PATTERN)


GLASS_CSS = """
:root { --glass:rgba(255,255,255,.62); --glass-strong:rgba(255,255,255,.82); --glass-line:rgba(255,255,255,.75);
        --shadow-1:0 1px 2px rgba(15,76,58,.06), 0 8px 24px rgba(15,76,58,.08); --shadow-2:0 2px 4px rgba(15,76,58,.08), 0 18px 44px rgba(15,76,58,.16); }
.gradio-container, body.icv-page { background:
    radial-gradient(900px 520px at 88% -8%, rgba(226,192,110,.34), transparent 60%),
    radial-gradient(760px 520px at 6% 4%, rgba(23,105,79,.20), transparent 62%),
    linear-gradient(180deg,#FBF6EA 0%,#F3EBD3 100%) !important; background-attachment:fixed !important; }
/* RTL safety: long words, URLs and mixed Latin/digit runs wrap inside their box instead of spilling out */
.icv, .icv * { box-sizing:border-box; min-width:0; }
.icv p, .icv span, .icv b, .icv label, .icv summary, .icv button, .icv .badge, .icv .tile { overflow-wrap:anywhere; }
.icv .quote, .icv .diff, .icv .final-text, .icv .highlight p, .icv .generated p { unicode-bidi:plaintext; text-align:start; }
.icv .hero { position:relative; overflow:hidden; border-bottom:0; border:1px solid rgba(226,192,110,.45);
  background:linear-gradient(120deg,#0B3B2D,#17694F 45%,#0F4C3A 70%,#1d7a5c); background-size:240% 240%; animation:icv-flow 16s ease-in-out infinite;
  box-shadow:var(--shadow-2); }
.icv .hero::before { content:""; position:absolute; inset:-40% -10% auto auto; width:60%; aspect-ratio:1; border-radius:50%;
  background:radial-gradient(circle, rgba(226,192,110,.38), transparent 65%); pointer-events:none; }
.icv .hero::after { content:""; position:absolute; inset:auto auto 0 0; width:100%; height:4px; background:linear-gradient(90deg,transparent,var(--gold-2),transparent); }
.icv .hero h1 { font-size:clamp(1.45rem,4.2vw,2.3rem); line-height:1.5; text-wrap:balance; position:relative; }
.icv .hero .tagline, .icv .hero .sub, .icv .flow { position:relative; }
.icv .flow span { backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px); }
@keyframes icv-flow { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
.icv .tile, .icv .highlight, .icv .generated, .icv .final, .icv .qcard, .icv .notice, .icv .export {
  background:var(--glass); border:1px solid var(--glass-line); box-shadow:var(--shadow-1);
  backdrop-filter:blur(14px) saturate(150%); -webkit-backdrop-filter:blur(14px) saturate(150%); transition:transform .25s ease, box-shadow .25s ease; }
.icv .qcard { border-inline-start:6px solid var(--line); }
.icv .qcard:hover, .icv .tile:hover { transform:translateY(-2px); box-shadow:var(--shadow-2); }
.icv .tile.verified { background:linear-gradient(160deg,rgba(234,246,239,.92),rgba(255,255,255,.6)); }
.icv .tile.mismatch { background:linear-gradient(160deg,rgba(253,236,234,.92),rgba(255,255,255,.6)); }
.icv .tile.review { background:linear-gradient(160deg,rgba(255,246,218,.95),rgba(255,255,255,.6)); }
.icv .final { background:linear-gradient(160deg,rgba(255,253,246,.92),rgba(250,240,208,.55)); border-color:rgba(184,145,47,.55); }
.icv .final-head { flex-wrap:wrap; gap:10px; }
.icv .qhead .badge { max-width:100%; white-space:normal; }
.icv .ind { grid-template-columns:minmax(96px,150px) minmax(0,1fr) 48px; }
.icv .bar span { background:linear-gradient(270deg,var(--green),var(--gold-2)); transition:width .6s ease; }
.icv .copy, .icv .dl { background:linear-gradient(135deg,var(--green),var(--green-2)); color:#FFF; border:0; border-radius:12px; padding:7px 18px;
  font-family:inherit; font-weight:600; cursor:pointer; box-shadow:0 4px 12px rgba(15,76,58,.25); transition:transform .15s ease, box-shadow .15s ease, filter .15s ease; }
.icv .copy:hover, .icv .dl:hover { transform:translateY(-1px); filter:brightness(1.08); box-shadow:0 8px 18px rgba(15,76,58,.3); }
.icv .copy:active, .icv .dl:active { transform:translateY(0); }
.icv .copy:focus-visible, .icv .dl:focus-visible { outline:3px solid rgba(184,145,47,.55); outline-offset:2px; }
.icv .copy.done { background:linear-gradient(135deg,#1B7A4B,#2a9d66); }
.icv .export { border-radius:16px; padding:12px 18px; margin:12px 0; }
.icv .export summary { cursor:pointer; color:var(--green); font-weight:700; }
.icv .dls { display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; }
/* toast */
#icv-toast { position:fixed; inset-inline:0; bottom:28px; margin-inline:auto; width:max-content; max-width:calc(100vw - 32px); z-index:99999;
  direction:rtl; text-align:center; font:600 1rem 'Cairo','Segoe UI',Tahoma,sans-serif; color:#FFF; padding:12px 24px; border-radius:999px;
  background:linear-gradient(135deg,rgba(15,76,58,.94),rgba(23,105,79,.94)); border:1px solid rgba(226,192,110,.6);
  box-shadow:0 14px 40px rgba(15,76,58,.35); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
  opacity:0; transform:translateY(18px) scale(.97); pointer-events:none; transition:opacity .28s ease, transform .28s cubic-bezier(.2,.9,.3,1.2); }
#icv-toast.show { opacity:1; transform:none; } #icv-toast.bad { background:linear-gradient(135deg,rgba(179,38,30,.95),rgba(214,69,58,.95)); }
/* skeleton loaders */
.skel { position:relative; overflow:hidden; background:rgba(15,76,58,.08); border-radius:10px; }
.skel::after { content:""; position:absolute; inset:0; transform:translateX(100%); animation:icv-shimmer 1.4s infinite;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.75),transparent); }
@keyframes icv-shimmer { 100% { transform:translateX(-100%); } }
.skel-card { background:var(--glass); border:1px solid var(--glass-line); border-radius:16px; padding:16px 18px; margin:12px 0; box-shadow:var(--shadow-1); }
.skel-line { height:14px; margin:10px 0; } .skel-line.w60 { width:60%; } .skel-line.w85 { width:85%; } .skel-line.w40 { width:40%; }
.skel-tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin:8px 0 14px; } .skel-tile { height:78px; border-radius:14px; }
.icv-spinner { width:18px; height:18px; border-radius:50%; border:3px solid rgba(15,76,58,.18); border-top-color:var(--gold); display:inline-block;
  vertical-align:middle; margin-inline-end:10px; animation:icv-spin .8s linear infinite; }
@keyframes icv-spin { to { transform:rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .icv .hero, .skel::after, .icv-spinner { animation:none; } .icv .qcard, .icv .tile, #icv-toast { transition:none; } }
@media (max-width:640px){ .icv .ind{grid-template-columns:minmax(84px,104px) minmax(0,1fr) 40px;} .icv .final-head{flex-direction:column; align-items:stretch;} .icv .copy{width:100%;} }
/* layout polish: centred, compact, professional */
.icv .notice, .icv .disclaimer, .icv .export { text-align:center; }
.icv .export .legend { max-width:640px; margin:8px auto 0; font-size:.9rem; line-height:1.9; color:var(--muted); }
.icv .dls { justify-content:center; }
.icv .dl { display:inline-flex; flex-direction:column; align-items:center; gap:2px; min-width:170px; }
.icv .dl .t { font-weight:700; } .icv .dl .s { font-size:.78rem; font-weight:500; opacity:.85; }
.icv .hero .flow { display:flex; justify-content:center; flex-wrap:wrap; gap:8px; }

/* ---- Amanah AI visual identity ---- */
.icv .hero { padding:38px 24px 30px; border-radius:26px; background:
  radial-gradient(900px 300px at 85% -10%, rgba(226,192,110,.28), transparent 60%),
  radial-gradient(700px 320px at 0% 110%, rgba(23,105,79,.9), transparent 70%),
  linear-gradient(135deg,#0A3B2C 0%,#0F4C3A 45%,#17694F 100%); box-shadow:0 18px 50px rgba(10,59,44,.28); }
.icv .hero .mark { width:60px; height:60px; margin-bottom:6px; filter:drop-shadow(0 4px 10px rgba(226,192,110,.45)); }
.icv .hero h1.brand { font-family:'Cairo','Segoe UI',sans-serif; font-weight:800; font-size:clamp(2.4rem,7vw,3.9rem); letter-spacing:.02em; direction:ltr;
  line-height:1.15; margin:.1rem 0 .2rem; background:linear-gradient(180deg,#FFFFFF 30%,#F3DFA6 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.icv .hero .tagline { font-size:clamp(1.05rem,2.6vw,1.45rem); font-weight:700; color:var(--gold-2); margin:.2rem 0 .8rem; }
.icv .hero .sub { max-width:760px; font-size:1.02rem; line-height:2; color:#E4EFEA; }
.icv .trust { list-style:none; display:flex; flex-wrap:wrap; justify-content:center; gap:10px; margin:20px 0 0; padding:0; position:relative; }
.icv .trust li { background:rgba(255,255,255,.1); border:1px solid rgba(226,192,110,.5); border-radius:14px; padding:8px 16px; color:#FFF; font-size:.92rem; backdrop-filter:blur(6px); }
.icv .trust li b { color:var(--gold-2); font-size:1.05rem; margin-inline-end:6px; }
.icv .tile { box-shadow:0 6px 18px rgba(15,76,58,.07); transition:transform .2s; } .icv .tile:hover { transform:translateY(-2px); }
.icv .qcard { box-shadow:0 8px 26px rgba(15,76,58,.09); }
"""
CSS += GLASS_CSS

SKELETON = (
    '<div class="icv" aria-busy="true" aria-label="جارٍ التحقق">'
    '<div class="skel-tiles">' + '<div class="skel skel-tile"></div>' * 4 + '</div>'
    + ('<div class="skel-card"><div class="skel skel-line w40"></div><div class="skel skel-line w85"></div>'
       '<div class="skel skel-line"></div><div class="skel skel-line w60"></div></div>') * 2
    + '</div>'
)

COPY_JS = r"""
(function () {
  if (window.icvCopy) return;
  function toast(message, bad) {
    var t = document.getElementById('icv-toast');
    if (!t) { t = document.createElement('div'); t.id = 'icv-toast'; t.setAttribute('role', 'status'); t.setAttribute('aria-live', 'polite'); document.body.appendChild(t); }
    t.textContent = message; t.className = 'show' + (bad ? ' bad' : '');
    clearTimeout(window.__icvToast); window.__icvToast = setTimeout(function () { t.className = bad ? 'bad' : ''; }, 2400);
  }
  function legacyCopy(text) {
    var area = document.createElement('textarea'); area.value = text; area.setAttribute('readonly', '');
    area.style.cssText = 'position:fixed;top:0;left:0;opacity:0;pointer-events:none'; document.body.appendChild(area);
    area.select(); area.setSelectionRange(0, text.length); var ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    document.body.removeChild(area); return ok;
  }
  window.icvToast = toast;
  window.icvCopy = function (button) {
    var text = button.getAttribute('data-text') || '';
    function finish(ok) {
      toast(ok ? '__COPIED__' : 'تعذّر النسخ، حدّد النص وانسخه يدويًا', !ok);
      if (ok) { button.classList.add('done'); setTimeout(function () { button.classList.remove('done'); }, 1600); }
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(function () { finish(true); }, function () { finish(legacyCopy(text)); });
    } else { finish(legacyCopy(text)); }
  };
  window.icvDownload = function (button) {
    var blob = new Blob([button.getAttribute('data-text') || ''], { type: 'text/tab-separated-values;charset=utf-8' });
    var link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = button.getAttribute('data-name') || 'results.tsv';
    document.body.appendChild(link); link.click(); document.body.removeChild(link); setTimeout(function () { URL.revokeObjectURL(link.href); }, 1000);
  };
})();
""".replace("__COPIED__", COPIED_MESSAGE)
