from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

import ui
from camelbert_adapter import analyze_hybrid, entities_to_spans, query_hosted_model
from llm_client import LLMError, generate, sanitize_answer
from verifier import MAX_INPUT_CHARS, IslamicContentVerifier

logger = logging.getLogger(__name__)

EXAMPLES_PATH = Path(__file__).resolve().parent / "demo" / "examples.json"

_pipeline: Optional[IslamicContentVerifier] = None
_lock = threading.Lock()


def get_pipeline() -> IslamicContentVerifier:
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = IslamicContentVerifier()
        return _pipeline


def warm_in_background() -> None:
    threading.Thread(target=lambda: get_pipeline().retriever.warm(), daemon=True).start()


def load_examples(path: Path = EXAMPLES_PATH) -> List[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not load demo examples from %s", path)
        return []


def _analyze(text: str, entities_json: str = "") -> dict:
    pipeline = get_pipeline()
    model_spans: list = []
    try:
        if entities_json:
            model_spans = entities_to_spans(text, json.loads(entities_json))
        elif os.environ.get("ICV_HF_MODEL", "").strip():
            model_spans = query_hosted_model(text, os.environ["ICV_HF_MODEL"].strip(), os.environ.get("HF_TOKEN", ""))
    except (RuntimeError, ValueError, TypeError):
        logger.warning("Hosted model unavailable; using the bundled detector only")
        model_spans = []
    return analyze_hybrid(pipeline, text, model_spans)


def verify_text(text: str, entities_json: str = "") -> str:
    if not text or not text.strip():
        return ui.render_message("الرجاء إدخال نص للتحقق منه.", "warn")
    try:
        return ui.render_results(_analyze(text, entities_json))
    except ValueError:
        return ui.render_message(f"النص طويل جدًا (الحد الأقصى {MAX_INPUT_CHARS} حرف).", "warn")
    except Exception:
        logger.exception("Verification failed")
        return ui.render_message("حدث خطأ غير متوقع أثناء التحقق.", "bad")


def verify_generated_answer(answer: str, entities_json: str = "") -> str:
    try:
        answer = sanitize_answer(answer)
        return ui.render_results(_analyze(answer, entities_json), generated_answer=answer)
    except ValueError:
        return ui.render_message(f"الإجابة طويلة جدًا (الحد الأقصى {MAX_INPUT_CHARS} حرف).", "warn")
    except Exception:
        logger.exception("Verification of the generated answer failed")
        return ui.render_message("تعذّر التحقق من إجابة النموذج.", "bad")


def ask_then_verify(prompt: str) -> str:
    try:
        answer = generate(None, prompt)
    except LLMError as exc:
        return ui.render_message(str(exc), "warn")
    return verify_generated_answer(answer)


def build_interface():
    import gradio as gr

    examples = load_examples()

    def next_example(index: int):
        if not examples:
            return "", 0
        return examples[index % len(examples)]["text"], (index + 1) % len(examples)

    samples = [e for e in examples if e.get("question")]

    def next_ask_example(index: int):
        if not samples:
            return "", 0
        return samples[index % len(samples)]["question"], (index + 1) % len(samples)

    def saved_answer_check(question: str):
        for sample in samples:
            if sample["question"] == question:
                return verify_generated_answer(sample["text"])
        return ui.render_message("اختر مثالًا أولًا.", "warn")

    theme = gr.themes.Base(primary_hue="emerald", neutral_hue="stone")
    with gr.Blocks(title=ui.APP_TITLE, css=ui.CSS, theme=theme, head=f"<script>{ui.COPY_JS}</script>") as demo:
        gr.HTML(ui.HERO)
        with gr.Tabs():
            with gr.Tab("تحقّق مباشر"):
                example_index = gr.State(0)
                text_input = gr.Textbox(label="النص المراد التحقق منه", lines=9, max_lines=24, placeholder=ui.PLACEHOLDER,
                                        rtl=True, elem_classes="input-area")
                with gr.Row():
                    verify_button = gr.Button("تحقّق من النص", variant="primary", scale=3)
                    example_button = gr.Button("جرّب مثالًا", variant="secondary", scale=2)
                results = gr.HTML(elem_classes="results")
                verify_button.click(verify_text, inputs=text_input, outputs=results)
                example_button.click(next_example, inputs=example_index, outputs=[text_input, example_index]).then(
                    verify_text, inputs=text_input, outputs=results)
            with gr.Tab("اسأل ثم تحقّق"):
                gr.HTML('<div class="icv"><div class="notice">اكتب سؤالًا، وسيجيب عنه النظام مباشرةً، ثم يفحص كل آية '
                        'وحديث ورد في الإجابة ويعرض الأخطاء والتصحيحات.</div></div>')
                prompt = gr.Textbox(label="سؤالك", lines=3, placeholder=ui.PROMPT_PLACEHOLDER, rtl=True, elem_classes="input-area")
                with gr.Row():
                    ask_button = gr.Button("اسأل ثم تحقّق", variant="primary", scale=3)
                    ask_example_button = gr.Button("جرّب مثالًا", variant="secondary", scale=2)
                answer_results = gr.HTML(elem_classes="results")
                ask_button.click(ask_then_verify, inputs=prompt, outputs=answer_results)
                ask_example_state = gr.State(0)
                ask_example_button.click(next_ask_example, inputs=ask_example_state, outputs=[prompt, ask_example_state]).then(
                    saved_answer_check, inputs=prompt, outputs=answer_results)
        gr.HTML(ui.DISCLAIMER)
    return demo


def _load_dotenv(path: Path = Path(__file__).resolve().parent / ".env") -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        name, sep, value = line.strip().partition("=")
        if sep and name and not name.startswith("#") and value.strip():
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _load_dotenv()
    get_pipeline()
    warm_in_background()
    build_interface().queue().launch(share=os.environ.get("ICV_SHARE") == "1")


if __name__ == "__main__":
    main()
