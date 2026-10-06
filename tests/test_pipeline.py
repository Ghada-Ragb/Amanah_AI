import json
import re
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  
import ui  
from alignment import align, allowed_gap, best_region  
from idgham import apply_idgham  
from benchmark_format import competition_rows, official_payload  
from camelbert_adapter import analyze_hybrid, entities_to_spans, merge_spans  
from llm_client import LLMError, LLMSettings, build_request, generate, parse_response  
from normalization import normalize_for_matching, normalize_strict, phonetic_key  
from similarity import compute_signals  
from verifier import MAX_INPUT_CHARS, IslamicContentVerifier  

HUD_112 = "فَاسْتَقِمْ كَمَا أُمِرْتَ وَمَنْ تَابَ مَعَكَ"


def quote(text):
    return f'قال الله تعالى: "{text}".'


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = IslamicContentVerifier()
        cls.kb = cls.pipe.retriever

    def status(self, text):
        return self.pipe.analyze(text)["spans"][0]


class TestNormalization(unittest.TestCase):
    def test_uthmani_and_modern_script_meet(self):
        self.assertEqual(normalize_for_matching("ٱلرَّحْمَـٰنِ"), normalize_for_matching("الرحمن"))
        self.assertEqual(normalize_for_matching("ﷲ"), "الله")
        self.assertEqual(normalize_strict("إِنَّ اللَّهَ"), "ان الله")

    def test_phonetic_key_folds_sound_alikes(self):
        self.assertEqual(phonetic_key("الصبر"), phonetic_key("السبر"))

    def test_idgham(self):
        self.assertIsInstance(apply_idgham("مِن رَّبِّهِمْ"), str)


class TestAlignment(unittest.TestCase):
    def test_substitution_is_detected(self):
        result = align("ثُمَّ اسْتَقِمْ كَمَا أُمِرْتَ وَمَن تَابَ مَعَكَ", HUD_112)
        self.assertFalse(result["exact"])
        self.assertTrue(result["near"])

    def test_reordering_is_detected(self):
        result = align("كَمَا أُمِرْتَ فَاسْتَقِمْ وَمَنْ تَابَ مَعَكَ", HUD_112)
        self.assertTrue(result["reordered"])
        self.assertFalse(result["exact"])

    def test_missing_diacritics_are_accepted_and_wrong_ones_noted(self):
        self.assertTrue(align("فاستقم كما أمرت ومن تاب معك", HUD_112)["exact"])
        wrong = align("فَاسْتَقَمْ كَمَا أُمِرْتَ وَمَنْ تَابَ مَعَكَ", HUD_112)
        self.assertTrue(wrong["exact"])
        self.assertEqual(len(wrong["diacritic_notes"]), 1)

    def test_partial_quote_is_not_penalised_for_length(self):
        full = "يَا أَيُّهَا الَّذِينَ آمَنُوا اسْتَعِينُوا بِالصَّبْرِ وَالصَّلَاةِ إِنَّ اللَّهَ مَعَ الصَّابِرِينَ"
        part = "إِنَّ اللَّهَ مَعَ الصَّابِرِينَ"
        self.assertTrue(align(part, full)["exact"])
        self.assertEqual(best_region(part, full), part)
        self.assertGreaterEqual(compute_signals(part, full, "Ayah")["composite"], 0.95)
        self.assertLess(compute_signals(part, full, "Ayah", local=False)["composite"], 0.8)

    def test_spelling_variants_are_not_errors_but_real_words_are(self):
        vocab = {"قتل", "قاتل", "اسحاق", "يا", "ايها"}
        self.assertTrue(align("ياأيها", "يا أيها", vocab)["exact"])
        self.assertTrue(align("اسحق", "إسحاق", vocab)["exact"])
        self.assertFalse(align("قتل", "قاتل", vocab)["exact"])

    def test_gap_scales_with_length(self):
        self.assertEqual(allowed_gap(3), 0)
        self.assertGreater(allowed_gap(20), allowed_gap(5))


class TestRetrieval(Base):
    def test_indexes_and_lazy_hadith(self):
        self.assertEqual(len(self.kb.quran), 6236)
        self.assertEqual(len(self.kb.quran_by_surah), 114)
        self.kb.warm()
        self.assertGreater(len(self.kb.hadith), 30000)

    def test_bm25_finds_partial_ayah_and_hadith(self):
        words = self.kb.quran[self.kb.quran_by_surah[17][23]]["text"].split()[:6]
        found = {(c["surah_id"], c["ayah_id"]) for c in self.kb.search_quran_ayahs(" ".join(words))}
        self.assertIn((17, 23), found)
        top = self.kb.search_hadith("إنما الأعمال بالنيات وإنما لكل امرئ ما نوى", top_k=1)[0]
        self.assertIn("صحيح البخاري", top["title"])


class TestVerification(Base):
    def test_exact_quotation_is_verified(self):
        span = self.status(quote(HUD_112))
        self.assertEqual(span["status"], "VERIFIED")
        self.assertEqual(span["group"], "verified")

    def test_substituted_word_gets_source_correction(self):
        span = self.status(quote("ثُمَّ اسْتَقِمْ كَمَا أُمِرْتَ وَمَنْ تَابَ مَعَكَ"))
        self.assertEqual(span["status"], "CORRECTED")
        self.assertEqual(span["correction"]["display_text"], HUD_112)

    def test_swapped_order_is_flagged(self):
        span = self.status(quote("كَمَا أُمِرْتَ فَاسْتَقِمْ وَمَنْ تَابَ مَعَكَ"))
        self.assertEqual(span["status"], "CORRECTED")
        self.assertTrue(span["reason"]["reordered"])

    def test_partial_citation_is_verified(self):
        self.assertEqual(self.status(quote("وَمَنْ تَابَ مَعَكَ وَلَا تَطْغَوْا"))["status"], "VERIFIED")

    def test_wrong_diacritic_keeps_status_but_adds_note(self):
        span = self.status(quote("فَاسْتَقَمْ كَمَا أُمِرْتَ وَمَنْ تَابَ مَعَكَ"))
        self.assertEqual(span["status"], "VERIFIED")
        self.assertIn("diacritic_conflict", [n["code"] for n in span["notes"]])

    def test_fabricated_text_is_never_verified_or_corrected(self):
        span = self.status(quote("ثم اجتهدوا في العمل وأكثروا من الصبر والصلاة دائما"))
        self.assertNotEqual(span["status"], "VERIFIED")
        self.assertIsNone(span["correction"])

    def test_hadith_chimera_goes_to_review_and_is_never_corrected(self):
        a = self.kb.search_hadith("إنما الأعمال بالنيات", top_k=1)[0]["text"].split()
        b = self.kb.search_hadith("الراحمون يرحمهم الرحمن", top_k=1)[0]["text"].split()
        mixed = " ".join(a[: len(a) // 2] + b[len(b) // 2:])
        span = self.status(f'قال النبي ﷺ: "{mixed}".')
        self.assertNotEqual(span["status"], "VERIFIED")
        self.assertIsNone(span["correction"])

    def test_empty_and_invalid_input(self):
        self.assertEqual(self.pipe.analyze("")["summary"]["n_spans"], 0)
        with self.assertRaises(TypeError):
            self.pipe.analyze(None)
        with self.assertRaises(ValueError):
            self.pipe.analyze("ا" * (MAX_INPUT_CHARS + 1))


class TestUnannouncedDetection(Base):
    def test_embedded_quranic_and_hadith_text_is_found_without_marks(self):
        text = ("وهنا نتذكر أن العبد عليه أن يثبت وألا يتراجع، فاستقم كما أمرت ومن تاب معك ولا تطغوا، وهذا من أعظم ما يعين. "
                "وكما جاء في الأثر إنما الأعمال بالنيات وإنما لكل امرئ ما نوى فمن كانت هجرته إلى دنيا يصيبها أو إلى امرأة ينكحها.")
        result = self.pipe.analyze(text)
        self.assertEqual([s["type"] for s in result["spans"]], ["Ayah", "Hadith"])
        self.assertTrue(all(s["detection"]["backend"] == "scan" for s in result["spans"]))
        self.assertTrue(all(s["status"] == "VERIFIED" for s in result["spans"]))

    def test_one_changed_word_is_still_found_and_flagged(self):
        text = "ومن المعاني التي نحتاجها اليوم قوله ثم استقم كما أمرت ومن تاب معك ولا تطغوا وكل هذا خير للناس."
        spans = self.pipe.analyze(text)["spans"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["status"], "CORRECTED")

    def test_ordinary_prose_is_left_alone(self):
        prose = ["الصبر مفتاح الفرج، والإنسان الذي يتحلى بالصبر يستطيع أن يواجه مشكلات الحياة بهدوء وثبات وعزيمة.",
                 "تعلن الجامعة عن فتح باب التسجيل للفصل الدراسي القادم، ويمكن للطلاب مراجعة الموقع لمعرفة المواعيد."]
        for text in prose:
            self.assertEqual(self.pipe.analyze(text)["spans"], [])

    def test_delimited_quotation_keeps_priority(self):
        span = self.pipe.analyze(quote(HUD_112))["spans"][0]
        self.assertEqual(span["detection"]["backend"], "rules")


class TestLLMClient(unittest.TestCase):
    def test_request_shape_has_no_model_choice_and_no_key_in_body(self):
        url, headers, body = build_request(LLMSettings("KEY"), "سؤال")
        self.assertTrue(url.endswith("/chat/completions"))
        self.assertEqual(headers["Authorization"], "Bearer KEY")
        self.assertEqual(body["model"], "gpt-4o-mini")
        self.assertNotIn("KEY", json.dumps(body))

    def test_parsing_and_errors(self):
        self.assertEqual(parse_response({"choices": [{"message": {"content": " نص "}}]}), "نص")
        with self.assertRaises(LLMError):
            parse_response({})
        with self.assertRaises(LLMError):
            generate(LLMSettings(""), "سؤال")
        with self.assertRaises(LLMError):
            generate(LLMSettings("k"), "  ")
        with self.assertRaises(LLMError):
            generate(LLMSettings("k"), "ا" * 2000)

    def test_round_trip_against_a_local_server(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                if self.headers.get("Authorization") == "Bearer bad":
                    self.send_response(401)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "إجابة"}}]}).encode())

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        self.assertEqual(generate(LLMSettings("k", base_url=base), "سؤال"), "إجابة")
        with self.assertRaises(LLMError):
            generate(LLMSettings("bad", base_url=base), "سؤال")
        server.shutdown()


class TestNoSecretsShipped(unittest.TestCase):

    ROOT = Path(__file__).resolve().parents[1]

    def test_no_api_key_pattern_in_source_files(self):
        pattern = re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")
        for path in list(self.ROOT.glob("*.py")) + list(self.ROOT.glob("*.md")) + list((self.ROOT / "functions").rglob("*.js")) + list((self.ROOT / "server").glob("*.js")) + list((self.ROOT / "worker").glob("*")) \
                + list((self.ROOT / "tests").glob("*.py")):
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), path.name)

    def test_proxy_uses_the_same_system_prompt_and_reads_the_key_from_env(self):
        js = (self.ROOT / "server" / "ask-core.js").read_text(encoding="utf-8")
        from llm_client import SYSTEM_PROMPT
        self.assertIn(json.dumps(SYSTEM_PROMPT, ensure_ascii=False), js)
        self.assertIn("env.OPENAI_API_KEY", js)

    def test_built_page_has_no_provider_model_or_key_inputs(self):
        import build_static_space
        page = build_static_space.build_index("test")
        for needle in ('id="provider"', 'id="model"', 'id="key"', "api.openai.com", "x-goog-api-key", "Authorization"):
            self.assertNotIn(needle, page)
        self.assertIn(ui.APP_TITLE, page)
        self.assertNotIn("مدقق المحتوى الإسلامي", page)


class TestDetectionWithoutPrefixes(Base):
    def test_delimited_quotations_are_typed_by_the_corpora_not_by_a_prefix(self):
        for text, label in (('"%s"' % HUD_112, "Ayah"),
                            ('ورد: "إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ ، وَإِنَّمَا لِكُلِّ امْرِئٍ مَا نَوَى" وهذا مهم.', "Hadith"),
                            ('قال رسول الله ﷺ: "%s"' % HUD_112, "Ayah")):   # a misleading prefix does not win
            spans = self.pipe.analyze(text)["spans"]
            self.assertEqual([s["type"] for s in spans], [label], text)

    def test_ordinary_quoted_prose_is_not_reported(self):
        self.assertEqual(self.pipe.analyze('قال "مرحبا بكم في موقعنا الجديد اليوم" ثم انصرف.')["spans"], [])

    def test_trigger_mode_can_be_restored_for_ablation(self):
        legacy = IslamicContentVerifier(retriever=self.kb, decouple_triggers=False)
        self.assertEqual([s["detection"]["backend"] for s in legacy.analyze('"%s"' % HUD_112)["spans"]], ["scan"])
        self.assertEqual([s["detection"]["backend"] for s in self.pipe.analyze('"%s"' % HUD_112)["spans"]], ["rules"])


class TestOfficialFormat(Base):
    def test_rows_follow_the_three_file_layouts(self):
        text = quote("ثُمَّ اسْتَقِمْ كَمَا أُمِرْتَ وَمَنْ تَابَ مَعَكَ")
        a, b, c = competition_rows(self.pipe.analyze(text), "Q7")
        self.assertEqual(a[0][0], "Q7")
        self.assertEqual(a[0][3], "Ayah")
        self.assertEqual(b[0], ("Q7_1", "Incorrect"))
        self.assertEqual(c[0][0], f"Q7_{a[0][1]}_{a[0][2]}")
        self.assertIn("فَاسْتَقِمْ", c[0][1])

    def test_no_spans_and_tsv_payload(self):
        a, b, c = competition_rows(self.pipe.analyze("نص عادي لا اقتباس فيه."), "Q1")
        self.assertEqual((a, b, c), ([("Q1", 0, 0, "No_Spans")], [], []))
        payload = official_payload(self.pipe.analyze(quote(HUD_112)), "Q2")
        self.assertEqual(set(payload), {"task1A_predictions.tsv", "task1B_predictions.tsv", "task1C_predictions.tsv"})
        self.assertTrue(payload["task1B_predictions.tsv"].startswith("Q2_1\tCorrect"))


class TestCamelbertAdapter(Base):
    def test_aggregated_and_raw_entities_become_spans(self):
        text = "قال الله فاستقم كما أمرت ثم قال"
        start = text.index("فاستقم")
        end = start + len("فاستقم كما أمرت")
        aggregated = [{"entity_group": "Ayah", "score": 0.97, "start": start, "end": end}]
        self.assertEqual(entities_to_spans(text, aggregated), [{"label": "Ayah", "start": start, "end": end, "score": 0.97}])
        raw = [{"entity": "B-Ayah", "score": 0.9, "start": start, "end": start + 6},
               {"entity": "I-Ayah", "score": 0.9, "start": start + 7, "end": end}]
        self.assertEqual([(s["start"], s["end"]) for s in entities_to_spans(text, raw)], [(start, end)])
        self.assertEqual(entities_to_spans(text, [{"entity_group": "Ayah", "score": 0.2, "start": start, "end": end}]), [])
        self.assertEqual(entities_to_spans(text, [{"entity_group": "O", "score": 0.99, "start": 0, "end": 3}]), [])

    def test_without_a_model_the_result_is_the_bundled_pipeline(self):
        text = quote(HUD_112)
        result = analyze_hybrid(self.pipe, text, [])
        self.assertEqual(result["spans"][0]["status"], "VERIFIED")
        self.assertEqual(result["spans"][0]["detection"]["backend"], self.pipe.analyze(text)["spans"][0]["detection"]["backend"])
        self.assertNotIn("محاكاة", ui.render_results(result))

    def test_model_spans_fill_gaps_but_never_override_the_detector(self):
        text = "هذه الآية فاستقم كما أمرت ومن تاب معك صحيحة"
        start = text.index("فاستقم")
        model = [{"label": "Ayah", "start": start, "end": start + len("فاستقم كما أمرت ومن تاب معك"), "score": 0.9}]
        merged = merge_spans(text, [], model)
        self.assertEqual([s.source for s in merged], ["camelbert"])
        detected = self.pipe.detect(quote(HUD_112))
        overlapping = [{"label": "Hadith", "start": detected[0].start, "end": detected[0].end, "score": 0.99}]
        self.assertEqual(merge_spans(quote(HUD_112), detected, overlapping), detected)

    def test_browser_entry_point(self):
        text = "هذه الآية فاستقم كما أمرت ومن تاب معك صحيحة"
        start = text.index("فاستقم")
        entities = json.dumps([{"entity_group": "Ayah", "score": 0.95, "start": start, "end": start + len("فاستقم كما أمرت ومن تاب معك")}])
        self.assertIn("موثّق", app.verify_text(text, entities))
        self.assertIn("generated", app.verify_generated_answer(text, entities))
        self.assertIn("موثّق", app.verify_text(text, "not json"))   


class TestScreenshotRegressions(Base):
    def test_whole_ayah_of_common_words_is_found(self):
        result = self.pipe.analyze("قل هو الله احد")
        self.assertEqual([(s["type"], s["status"]) for s in result["spans"]], [("Ayah", "VERIFIED")])

    def test_two_word_hadith_quote_is_listed_for_review_not_dropped(self):
        text = 'قال رسول الله ﷺ: "...الراحم عطية"، رواه أبو داود.'
        spans = self.pipe.analyze(text)["spans"]
        self.assertEqual([(s["type"], s["status"], s["reason"]["code"]) for s in spans], [("Hadith", "HUMAN_REVIEW", "too_short")])
        self.assertIsNone(spans[0]["correction"])
        self.assertIn("قصير", ui.render_results(self.pipe.analyze(text)))

    def test_complete_two_word_ayah_is_verified(self):
        spans = self.pipe.analyze('قال الله تعالى: "وَثِيَابَكَ فَطَهِّرْ"')["spans"]
        self.assertEqual([(s["type"], s["status"]) for s in spans], [("Ayah", "VERIFIED")])

    def test_one_changed_word_in_a_hadith_is_not_verified(self):
        text = 'قال رسول الله ﷺ: "مَنْ كَانَ يُؤْمِنُ بِاللَّهِ وَالْيَوْمِ الْآخِرِ فَلْيَقُلْ شَرًّا أَوْ لِيَصْمُتْ"'
        span = self.pipe.analyze(text)["spans"][0]
        self.assertEqual((span["status"], span["reason"]["code"], span["verification"]["verdict"]), ("HUMAN_REVIEW", "hadith_altered", "Incorrect"))
        self.assertIsNone(span["correction"])

    def test_an_extra_or_missing_word_in_a_hadith_is_not_verified(self):
        for text in ('قال رسول الله ﷺ: "مَنْ كَانَ يُؤْمِنُ بِاللَّهِ وَالْيَوْمِ الْآخِرِ فَلْيَقُلْ خَيْرًا كَثِيرًا أَوْ لِيَصْمُتْ"',
                     'قال رسول الله ﷺ: "مَنْ كَانَ يُؤْمِنُ بِاللَّهِ وَالْيَوْمِ الْآخِرِ فَلْيَقُلْ أَوْ لِيَصْمُتْ"'):
            self.assertEqual(self.pipe.analyze(text)["spans"][0]["status"], "HUMAN_REVIEW", text)

    def test_misattribution_is_pointed_out(self):
        span = self.pipe.analyze('قال رسول الله ﷺ: "إِنَّ مَعَ الْعُسْرِ يُسْرًا"')["spans"][0]
        self.assertTrue(any(n.get("misattributed") for n in span["notes"]))
        span = self.pipe.analyze('قال الله تعالى: "الطُّهُورُ شَطْرُ الْإِيمَانِ"')["spans"][0]
        self.assertEqual(span["status"], "UNSUPPORTED")
        self.assertTrue(any(n["code"] == "is_hadith" for n in span["notes"]))

    def test_same_words_are_never_reported_twice(self):
        text = "قال رسول الله ﷺ:\nمَنْ كَانَ يُؤْمِنُ بِاللَّهِ وَالْيَوْمِ الْآخِرِ فَلْيَقُلْ خَيْرًا أَوْ لِيَصْمُتْ\nرواه البخاري"
        spans = self.pipe.analyze(text)["spans"]
        for i, a in enumerate(spans):
            for b in spans[i + 1:]:
                self.assertTrue(a["end"] <= b["start"] or b["end"] <= a["start"])

    def test_scenario_examples_cover_every_combination(self):
        by_id = {e["id"]: e for e in app.load_examples()}
        expected = {
            "scenario_all_correct_ayahs": {"VERIFIED"},
            "scenario_all_wrong_ayahs": {"CORRECTED"},
            "scenario_all_correct_hadiths": {"VERIFIED"},
            "scenario_all_wrong_hadiths": {"UNSUPPORTED"},
            "scenario_wrong_ayahs_correct_hadiths": {"CORRECTED", "VERIFIED"},
            "scenario_correct_ayahs_wrong_hadiths": {"VERIFIED", "UNSUPPORTED"},
        }
        for key, statuses in expected.items():
            self.assertTrue(by_id[key]["question"])
            self.assertEqual({s["status"] for s in self.pipe.analyze(by_id[key]["text"])["spans"]}, statuses, key)

    def test_results_have_no_export_section(self):
        self.assertNotIn("task1A", ui.render_results(self.pipe.analyze(quote(HUD_112))))

    def test_two_words_without_an_introduction_are_ignored(self):
        self.assertEqual(self.pipe.analyze('كتب "الراحم عطية" على الورقة')["spans"], [])

    def test_foreign_scripts_are_removed_from_answers(self):
        from llm_client import sanitize_answer
        self.assertEqual(sanitize_answer("الأمم... 不是، قال رسول الله"), "الأمم... قال رسول الله")
        self.assertEqual(sanitize_answer("نص عربي سليم، كما هو."), "نص عربي سليم، كما هو.")


class TestInterface(Base):
    def test_examples_match_observed_statuses_and_render(self):
        for example in app.load_examples():
            result = self.pipe.analyze(example["text"])
            self.assertEqual([s["status"] for s in result["spans"]], example["observed_statuses"], example["id"])
            ui.render_results(result)

    def test_output_is_arabic_only(self):
        for example in app.load_examples():
            visible = re.sub(r"<[^>]+>", " ", ui.render_results(self.pipe.analyze(example["text"])))
            visible = re.sub(r"&\w+;|&#\d+;", " ", visible)
            self.assertEqual(re.findall(r"[A-Za-z]{2,}", visible), [], example["id"])

    def test_verified_cards_contain_no_red(self):
        html = ui.render_results(self.pipe.analyze(quote(HUD_112)))
        self.assertNotIn("w-extra", html)
        self.assertNotIn('class="qcard mismatch"', html)
        self.assertNotIn('badge st mismatch', html)

    def test_human_review_message_and_never_an_invented_replacement(self):
        example = next(e for e in app.load_examples() if e["id"] == "uncertain_hadith")
        html = ui.render_results(self.pipe.analyze(example["text"]))
        self.assertIn("الأدلة غير كافية للتصحيح الآلي", html)
        self.assertIn("المراجعة البشرية", html)
        self.assertNotIn("التصحيح المقترح من المصدر", html)

    def test_copy_button_uses_the_clipboard_helper_and_toast_message(self):
        html = ui.render_results(self.pipe.analyze(quote("ثُمَّ اسْتَقِمْ كَمَا أُمِرْتَ وَمَنْ تَابَ مَعَكَ")))
        self.assertIn("window.icvCopy(this)", html)
        self.assertIn("نسخ النص", html)
        self.assertEqual(ui.COPIED_MESSAGE, "تم نسخ النص بنجاح")
        self.assertIn(ui.COPIED_MESSAGE, ui.COPY_JS)
        self.assertIn("navigator.clipboard", ui.COPY_JS)
        self.assertIn("execCommand", ui.COPY_JS)   

    def test_branding(self):
        self.assertIn('<h1 class="brand">Amanah AI</h1>', ui.HERO)
        self.assertIn("نظام ذكي للتحقق من الاقتباسات القرآنية والحديثية", ui.HERO)
        self.assertEqual(ui.ENGLISH_TITLE, "AI-Powered Verification and Correction of Quranic and Prophetic Quotations")

    def test_ask_tab_calls_the_configured_client(self):
        self.assertIn("غير مهيّأة", app.ask_then_verify("سؤال"))   

    def test_handlers_never_raise(self):
        self.assertIn("notice", app.verify_text(""))
        self.assertIn("notice", app.verify_text("ا" * (MAX_INPUT_CHARS + 1)))
        self.assertIn("generated", app.verify_generated_answer(quote(HUD_112)))

    def test_mode_b_corrected_text(self):
        html = app.verify_generated_answer(quote("ثُمَّ اسْتَقِمْ كَمَا أُمِرْتَ وَمَنْ تَابَ مَعَكَ"))
        self.assertIn("النسخة المصحّحة", html)
        result = self.pipe.analyze(quote("ثُمَّ اسْتَقِمْ كَمَا أُمِرْتَ وَمَنْ تَابَ مَعَكَ"))
        self.assertEqual(ui.final_text(result), quote(HUD_112)[:-2] + '".')


if __name__ == "__main__":
    unittest.main()


class TestAntiHallucination(Base):
    def test_every_reference_and_correction_is_grounded_in_the_corpora(self):
        for example in app.load_examples():
            for span in self.pipe.analyze(example["text"])["spans"]:
                self.assertEqual(span["grounding"], "verified", example["id"])

    def test_1b_label_follows_the_final_decision(self):
        for example in app.load_examples():
            for span in self.pipe.analyze(example["text"])["spans"]:
                self.assertEqual(span["verification"]["verdict"], "Correct" if span["status"] == "VERIFIED" else "Incorrect", example["id"])

    def test_a_forged_correction_is_removed_and_sent_to_review(self):
        span = self.pipe.analyze('قال الله تعالى: "ثُمَّ اسْتَقِمْ كَمَا أُمِرْتَ وَمَنْ تَابَ مَعَكَ"')["spans"][0]
        self.assertEqual(span["status"], "CORRECTED")
        span["correction"]["display_text"] = "نص مختلق لا وجود له في المصحف"
        grounded = self.pipe._ground(span)
        self.assertEqual((grounded["status"], grounded["reason"]["code"], grounded["correction"]), ("HUMAN_REVIEW", "ungrounded", None))

    def test_a_reference_that_does_not_exist_is_dropped(self):
        span = self.pipe.analyze(quote(HUD_112))["spans"][0]
        span["evidence"]["source"] = {"type": "Quran", "surah_id": 999, "surah_name": "؟", "ayah_start": 1, "ayah_end": 1}
        self.assertEqual(self.pipe._ground(span)["status"], "HUMAN_REVIEW")

    def test_invented_quotations_are_never_verified(self):
        for text in ('قال الله تعالى: "إن الله يحب كل من يكتب الشعر في الليل ويقرأه على أصدقائه"',
                     'قال رسول الله ﷺ: "من قرأ هذه الكلمات ثلاثا غفر الله له كل ذنوبه وأدخله الجنة بغير حساب"',
                     'قال رسول الله ﷺ: "مَنْ أَكَلَ الْعَسَلَ مَعَ الثُّومِ سَبْعَةَ أَيَّامٍ شُفِيَ مِنْ كُلِّ دَاءٍ"'):
            for span in self.pipe.analyze(text)["spans"]:
                self.assertNotEqual(span["status"], "VERIFIED", text)
                self.assertIsNone(span["correction"], text) if span["type"] == "Hadith" else None


class TestStaticPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_static_space
        cls.html = build_static_space.build_index("testbuild")

    def test_ask_and_verify_is_the_default_page(self):
        self.assertLess(self.html.index('data-tab="ask"'), self.html.index('data-tab="direct"'))
        self.assertIn('<section id="tab-ask" class="form glass">', self.html)
        self.assertIn('<section id="tab-direct" class="form glass" hidden>', self.html)

    def test_branding_and_no_duplicate_downloads(self):
        self.assertIn("Amanah AI", self.html)
        self.assertNotIn('rel="preload" href="index/', self.html)
        self.assertNotIn('rel="prefetch"', self.html)
        self.assertNotIn("task1A", self.html)

    def test_no_engine_selector_and_no_sample_answer_substitution(self):
        self.assertNotIn('id="engine"', self.html)
        self.assertNotIn("عُرضت إجابة تجريبية محفوظة", self.html)

    def test_loading_indicator_is_a_small_spinner_without_a_ready_banner(self):
        self.assertIn('id="boot-text"', self.html)
        self.assertNotIn("النظام جاهز", self.html)
        self.assertNotIn('class="meter"', self.html)
        self.assertLess(self.html.index('id="tab-direct"'), self.html.index('id="boot"'))  


class TestColabNotebook(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent
    NB = ROOT / "colab_verification_pipeline" / "amanah_ai_verification_pipeline.ipynb"

    def test_notebook_modules_are_identical_to_the_repository_files(self):
        import json as _json
        notebook = _json.loads(self.NB.read_text(encoding="utf-8"))
        embedded = {}
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and source.startswith("%%writefile "):
                name, _, body = source.partition("\n")
                embedded[name.split(" ", 1)[1].strip()] = body
        self.assertIn("verifier.py", embedded)
        self.assertIn("demo/examples.json", embedded)
        for name, body in embedded.items():
            path = self.ROOT / (name if name != "stress_test.py" else "research/stress_test.py")
            self.assertEqual(body.strip(), path.read_text(encoding="utf-8").strip(), name + " is stale: run build_notebook.py")

    def test_every_code_cell_follows_a_markdown_cell(self):
        import json as _json
        cells = _json.loads(self.NB.read_text(encoding="utf-8"))["cells"]
        for previous, cell in zip(cells, cells[1:]):
            if cell["cell_type"] == "code" and not "".join(cell["source"]).startswith("%%writefile demo/"):
                self.assertEqual(previous["cell_type"], "markdown")
