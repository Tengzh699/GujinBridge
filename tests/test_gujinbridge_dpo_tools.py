# -*- coding: utf-8 -*-
import importlib.util
import sys
import unittest
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_gujinbridge_dpo", "tools/prepare_gujinbridge_dpo.py")
finalize = load_module("finalize_gujinbridge_dpo", "tools/finalize_gujinbridge_dpo.py")
export_prompts = load_module(
    "export_gujinbridge_dpo_prompts", "tools/export_gujinbridge_dpo_prompts.py"
)
attach_predictions = load_module(
    "attach_gujinbridge_dpo_predictions", "tools/attach_gujinbridge_dpo_predictions.py"
)
select_pairs = load_module(
    "select_gujinbridge_dpo_pairs", "tools/select_gujinbridge_dpo_pairs.py"
)
compare_evals = load_module(
    "compare_gujinbridge_evals", "tools/compare_gujinbridge_evals.py"
)
analyze_errors = load_module(
    "analyze_gujinbridge_errors", "tools/analyze_gujinbridge_errors.py"
)
build_v6 = load_module(
    "build_gujinbridge_v6_curriculum", "tools/build_gujinbridge_v6_curriculum.py"
)
prepare_grpo = load_module(
    "prepare_gujinbridge_grpo", "tools/prepare_gujinbridge_grpo.py"
)
gujinbridge_grpo = load_module(
    "gujinbridge_grpo", "training/gujinbridge_grpo.py"
)
export_task_eval = load_module(
    "export_gujinbridge_task_eval", "tools/export_gujinbridge_task_eval.py"
)


class PrepareDPOTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "id": "one",
            "task": "c2m",
            "source": "论语",
            "category": "经",
            "conversations": [
                {"from": "system", "value": "系统提示"},
                {"from": "human", "value": "翻译：学而时习之"},
                {"from": "gpt", "value": "学习之后经常温习它。"},
            ],
        }

    def test_extract_prompt_excludes_final_answer(self):
        prompt, answer = prepare.extract_prompt_and_answer(self.record)
        self.assertEqual(len(prompt), 2)
        self.assertEqual(prompt[-1]["from"], "human")
        self.assertEqual(answer, "学习之后经常温习它。")

    def test_build_reviewed_pair_uses_corrected_as_chosen(self):
        correction = {
            "id": "one",
            "split": "train",
            "confidence": "high",
            "original_reference": "学习并练习。",
            "approved_reference": "学习之后经常温习它。",
            "reviewer": "reviewer",
        }
        pairs, skipped = prepare.build_reviewed_pairs({"one": self.record}, [correction])
        self.assertEqual(skipped, {})
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen"], correction["approved_reference"])
        self.assertEqual(pairs[0]["rejected"], correction["original_reference"])
        self.assertNotIn({"from": "gpt", "value": pairs[0]["chosen"]}, pairs[0]["conversations"])

    def test_generation_queue_excludes_reviewed_ids(self):
        other = dict(self.record)
        other["id"] = "two"
        selected = prepare.reservoir_candidates(
            [self.record, other], {"c2m": 10}, {"one"}, seed=42
        )
        queue = prepare.make_generation_queue(selected)
        self.assertEqual([item["source_id"] for item in queue], ["two"])
        self.assertEqual(queue[0]["review_status"], "pending_generation")


class FinalizeDPOTests(unittest.TestCase):
    def test_validate_pair(self):
        record = {
            "conversations": [{"from": "human", "value": "翻译"}],
            "chosen": "正确答案",
            "rejected": "错误答案",
        }
        self.assertEqual(finalize.validate_pair(record), "accepted")
        record["rejected"] = "正确答案"
        self.assertEqual(finalize.validate_pair(record), "identical_preference")

    def test_split_keeps_same_prompt_together(self):
        conversations = [{"from": "human", "value": "同一问题"}]
        pairs = [
            {"id": "a", "conversations": conversations, "chosen": "甲", "rejected": "乙"},
            {"id": "b", "conversations": conversations, "chosen": "甲", "rejected": "丙"},
            {
                "id": "c",
                "conversations": [{"from": "human", "value": "另一问题"}],
                "chosen": "丁",
                "rejected": "戊",
            },
        ]
        train, validation = finalize.split_by_prompt(pairs, validation_ratio=0.5, seed=42)
        train_ids = {item["id"] for item in train}
        validation_ids = {item["id"] for item in validation}
        self.assertTrue({"a", "b"}.issubset(train_ids) or {"a", "b"}.issubset(validation_ids))

    def test_cap_pairs_by_task_is_deterministic(self):
        pairs = [
            {
                "id": f"punct-{index}",
                "task": "punctuate",
                "conversations": [{"from": "human", "value": f"问题{index}"}],
                "chosen": f"正确{index}",
                "rejected": f"错误{index}",
            }
            for index in range(5)
        ]
        pairs.append(
            {
                "id": "c2m-keep",
                "task": "c2m",
                "conversations": [{"from": "human", "value": "翻译"}],
                "chosen": "正确",
                "rejected": "错误",
            }
        )
        first, removed = finalize.cap_pairs_by_task(pairs, {"punctuate": 2}, seed=42)
        second, _ = finalize.cap_pairs_by_task(pairs, {"punctuate": 2}, seed=42)
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertEqual(sum(item["task"] == "punctuate" for item in first), 2)
        self.assertIn("c2m-keep", {item["id"] for item in first})
        self.assertEqual(removed["punctuate"], 3)

    def test_parse_task_caps(self):
        self.assertEqual(
            finalize.parse_task_caps(["punctuate=160", "c2m=150"]),
            {"punctuate": 160, "c2m": 150},
        )
        with self.assertRaises(ValueError):
            finalize.parse_task_caps(["punctuate=0"])


class CandidatePipelineTests(unittest.TestCase):
    def test_export_extracts_system_and_user(self):
        record = {
            "id": "one",
            "conversations": [
                {"from": "system", "value": "系统"},
                {"from": "human", "value": "翻译：学而时习之"},
            ],
        }
        system, prompt = export_prompts.extract_system_and_prompt(record)
        self.assertEqual(system, "系统")
        self.assertEqual(prompt, "翻译：学而时习之")

    def test_attach_multiple_unique_predictions(self):
        queue = [
            {
                "id": "one",
                "conversations": [{"from": "human", "value": "翻译：学而时习之"}],
                "candidate_outputs": [],
            }
        ]
        predictions = [
            [{"Input": "翻译：学而时习之", "Output": "答案甲"}],
            [{"Input": "翻译：学而时习之", "Output": "答案乙"}],
        ]
        attached = attach_predictions.attach_predictions(queue, predictions)
        self.assertEqual(attached[0]["candidate_outputs"], ["答案甲", "答案乙"])
        self.assertEqual(attached[0]["review_status"], "pending_review")

    def test_selector_auto_approves_punctuation_text_change(self):
        records = [
            {
                "id": "punct-one",
                "task": "punctuate",
                "conversations": [{"from": "human", "value": "添加标点：\n\n学而时习之"}],
                "reference": "学而时习之。",
                "candidate_outputs": ["学而时习他。"],
            }
        ]
        approved, manual, skipped, counts = select_pairs.select_pairs(records)
        self.assertEqual(len(approved), 1)
        self.assertFalse(manual)
        self.assertFalse(skipped)
        self.assertIn("punctuation_text_changed", approved[0]["selection_reasons"])
        self.assertEqual(counts["auto_approved"], 1)

    def test_selector_keeps_ambiguous_translation_for_review(self):
        records = [
            {
                "id": "c2m-one",
                "task": "c2m",
                "conversations": [{"from": "human", "value": "翻译：\n\n学而时习之"}],
                "reference": "学习之后经常温习它。",
                "candidate_outputs": ["学习了以后时常复习。"],
            }
        ]
        approved, manual, skipped, _ = select_pairs.select_pairs(records)
        self.assertFalse(approved)
        self.assertEqual(len(manual), 1)
        self.assertFalse(skipped)

    def test_selector_does_not_auto_approve_translation_source_echo(self):
        records = [
            {
                "id": "c2m-echo",
                "task": "c2m",
                "conversations": [{"from": "human", "value": "翻译：\n\n学而时习之"}],
                "reference": "学习之后经常温习它。",
                "candidate_outputs": ["学而时习之"],
            }
        ]
        approved, manual, skipped, _ = select_pairs.select_pairs(records)
        self.assertFalse(approved)
        self.assertEqual(len(manual), 1)
        self.assertFalse(skipped)
        self.assertIn("source_echo_without_translation", manual[0]["review_flags"])

    def test_selector_requires_punctuation_reference_to_preserve_source(self):
        records = [
            {
                "id": "bad-reference",
                "task": "punctuate",
                "conversations": [{"from": "human", "value": "添加标点：\n\n学而时习之"}],
                "reference": "学而时习他。",
                "candidate_outputs": ["学而时习者。"],
            }
        ]
        approved, manual, skipped, _ = select_pairs.select_pairs(records)
        self.assertFalse(approved)
        self.assertEqual(len(manual), 1)
        self.assertFalse(skipped)
        self.assertIn("reference_text_changed", manual[0]["review_flags"])


class CompareEvalTests(unittest.TestCase):
    def test_compare_reports_pairwise_win_and_punctuation_preservation(self):
        references = [
            {
                "id": "one",
                "input": "给古文加标点： 学而时习之",
                "reference": "学而时习之。",
                "task": "punctuate",
            }
        ]
        baseline = defaultdict(deque, {"给古文加标点：学而时习之": deque(["学而时习他。"])})
        candidate = defaultdict(deque, {"给古文加标点：学而时习之": deque(["学而时习之。"])})
        report = compare_evals.compare(references, baseline, candidate)
        self.assertEqual(report["overall"]["pairwise_char_f1"]["wins"], 1)
        self.assertEqual(report["punctuation_text_preservation"]["baseline_preserved"], 0)
        self.assertEqual(report["punctuation_text_preservation"]["candidate_preserved"], 1)
        self.assertEqual(report["punctuation_quality"]["candidate_boundary_f1"], 1.0)

    def test_compare_rejects_incomplete_predictions(self):
        references = [{"id": "one", "input": "问题", "reference": "答案", "task": "c2m"}]
        with self.assertRaises(ValueError):
            compare_evals.compare(references, defaultdict(deque), defaultdict(deque))


class ErrorAnalysisTests(unittest.TestCase):
    def test_classifies_source_echo_and_omission(self):
        metrics = analyze_errors.classify_output(
            "c2m", "学而时习之不亦说乎", "学习后时常复习，不也是很愉快吗？", "学而时习之不亦说乎"
        )
        self.assertIn("source_echo", metrics["flags"])
        self.assertIn("omission_risk", metrics["flags"])

    def test_classifies_punctuation_text_change(self):
        metrics = analyze_errors.classify_output(
            "punctuate", "学而时习之", "学而时习之。", "学而时习他。"
        )
        self.assertIn("punctuation_text_changed", metrics["flags"])

    def test_review_queue_prioritizes_dpo_regression(self):
        references = [
            {
                "id": "one",
                "input": "翻译：学而时习之不亦说乎",
                "reference": "学习后时常复习，不也是很愉快吗？",
                "task": "c2m",
            }
        ]
        key = analyze_errors.normalize(references[0]["input"])
        baseline = defaultdict(deque, {key: deque([references[0]["reference"]])})
        candidate = defaultdict(deque, {key: deque(["学而时习之不亦说乎"])})
        _, rows = analyze_errors.analyze(references, baseline, candidate)
        queue = analyze_errors.select_review_queue(rows, 10)
        self.assertEqual(queue[0]["id"], "one")
        self.assertIn("dpo_regression", queue[0]["comparison_flags"])
        self.assertEqual(queue[0]["review_status"], "pending")


class V6CurriculumTests(unittest.TestCase):
    @staticmethod
    def record(record_id, task, source, human, answer):
        return {
            "id": record_id,
            "task": task,
            "source": source,
            "conversations": [
                {"from": "human", "value": human},
                {"from": "gpt", "value": answer},
            ],
        }

    def test_difficulty_detects_translation_risks(self):
        record = self.record(
            "hard",
            "c2m",
            "史书甲",
            "翻译：五月壬子，王虽不许，群臣固请，于是乃从之。",
            "五月壬子这一天，君王虽然没有允许，但群臣坚持请求，于是最终听从了他们。",
        )
        score, reasons = build_v6.c2m_difficulty(record)
        self.assertGreaterEqual(score, 7)
        self.assertIn("date_or_number", reasons)
        self.assertIn("negation", reasons)
        self.assertIn("logical_relation", reasons)

    def test_curriculum_keeps_replay_tasks(self):
        rows = [
            self.record("c-hard", "c2m", "甲", "翻译：五月壬子，王不许，故群臣请。", "五月某日，君王没有同意，所以群臣请求。"),
            self.record("c-general", "c2m", "乙", "翻译：山高。", "山很高。"),
            self.record("m", "m2c", "丙", "改写：山很高。", "山高。"),
            self.record("p", "punctuate", "丁", "标点：学而时习之", "学而时习之。"),
        ]
        selected = build_v6.select_curriculum(rows, 1, 1, 1, 1, seed=7)
        self.assertEqual(Counter(row["task"] for row in selected), Counter({"c2m": 2, "m2c": 1, "punctuate": 1}))

    def test_isolation_rejects_signature_overlap(self):
        train = [self.record("a", "c2m", "甲", "翻译：学而时习之", "经常复习。")]
        validation = [self.record("b", "c2m", "乙", "翻译：温故而知新", "复习旧知识。")]
        test = [self.record("c", "c2m", "丙", "翻译：学而时习之", "经常复习。")]
        result = build_v6.validate_isolation(train, validation, test)
        self.assertFalse(result["passed"])
        self.assertEqual(result["overlaps"]["train_vs_test"]["prompt_answer_signature_overlap"], 1)


class GRPOPunctuationTests(unittest.TestCase):
    def test_prepare_rejects_reference_that_changes_source(self):
        record = {
            "id": "bad",
            "task": "punctuate",
            "source": "论语",
            "conversations": [
                {"from": "human", "value": "标点：学而时习之不亦说乎"},
                {"from": "gpt", "value": "学而时习他，不亦说乎？"},
            ],
        }
        converted, reason = prepare_grpo.convert_record(record, 300)
        self.assertIsNone(converted)
        self.assertEqual(reason, "reference_changes_source")

    def test_preservation_reward_prefers_exact_source(self):
        completions = [
            [{"role": "assistant", "content": "学而时习之，不亦说乎？"}],
            [{"role": "assistant", "content": "学而时习他，不亦说乎？"}],
        ]
        rewards = gujinbridge_grpo.preservation_reward(
            completions, ["学而时习之不亦说乎", "学而时习之不亦说乎"]
        )
        self.assertEqual(rewards[0], 1.0)
        self.assertLess(rewards[1], 1.0)

    def test_punctuation_boundary_reward(self):
        reference = "学而时习之，不亦说乎？"
        self.assertEqual(gujinbridge_grpo.boundary_f1(reference, reference), 1.0)
        self.assertLess(gujinbridge_grpo.boundary_f1("学而时习之不亦说乎。", reference), 1.0)

    def test_direct_format_rejects_reasoning_tags(self):
        rewards = gujinbridge_grpo.direct_format_reward(
            [
                [{"role": "assistant", "content": "学而时习之。"}],
                [{"role": "assistant", "content": "<think>分析</think><answer>学而时习之。</answer>"}],
            ]
        )
        self.assertEqual(rewards, [1.0, 0.0])

    def test_v2_preservation_guard_is_negative_for_any_source_change(self):
        completions = [
            [{"role": "assistant", "content": "学而时习之，不亦说乎？"}],
            [{"role": "assistant", "content": "学而时习他，不亦说乎？"}],
        ]
        rewards = gujinbridge_grpo.preservation_guard_reward(
            completions, ["学而时习之不亦说乎", "学而时习之不亦说乎"]
        )
        self.assertEqual(rewards, [1.0, -1.0])

    def test_v2_position_reward_gives_partial_credit_for_wrong_mark_type(self):
        reference = "学而时习之，不亦说乎？"
        wrong_types = "学而时习之；不亦说乎。"
        self.assertEqual(gujinbridge_grpo.position_boundary_f1(wrong_types, reference), 1.0)
        self.assertLess(gujinbridge_grpo.boundary_f1(wrong_types, reference), 1.0)

    def test_v2_boundary_reward_is_gated_by_source_preservation(self):
        completions = [[{"role": "assistant", "content": "学而时习他，不亦说乎？"}]]
        rewards = gujinbridge_grpo.typed_boundary_reward_v2(
            completions,
            ["学而时习之，不亦说乎？"],
            ["学而时习之不亦说乎"],
        )
        self.assertEqual(rewards, [0.0])


if __name__ == "__main__":
    unittest.main()
