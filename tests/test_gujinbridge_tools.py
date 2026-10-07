# -*- coding: utf-8 -*-
import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_gujinbridge_dataset", "tools/prepare_gujinbridge_dataset.py")
evaluate = load_module("evaluate_gujinbridge_predictions", "tools/evaluate_gujinbridge_predictions.py")
audit = load_module("audit_gujinbridge_dataset", "tools/audit_gujinbridge_dataset.py")
gold = load_module("export_gujinbridge_gold_candidates", "tools/export_gujinbridge_gold_candidates.py")
merge_reviews = load_module("merge_gujinbridge_reviews", "tools/merge_gujinbridge_reviews.py")
audit_filter = load_module("filter_gujinbridge_audit_flags", "tools/filter_gujinbridge_audit_flags.py")
audit_candidates = load_module("export_gujinbridge_audit_candidates", "tools/export_gujinbridge_audit_candidates.py")
audit_summary = load_module("summarize_gujinbridge_audit_reviews", "tools/summarize_gujinbridge_audit_reviews.py")


class PrepareDatasetTests(unittest.TestCase):
    def test_parse_task_limits(self):
        self.assertEqual(prepare.parse_task_limits("c2m=10,m2c=-1"), {"c2m": 10, "m2c": -1})

    def test_convert_record_and_filters(self):
        raw = {
            "id": "one",
            "task": "c2m",
            "instruction": "翻译",
            "input": "学而时习之",
            "output": "学习后时常温习",
            "source": "论语·学而",
            "category": "经",
        }
        converted, reason = prepare.convert_record(raw, "系统提示", {"c2m"})
        self.assertEqual(reason, "accepted")
        self.assertEqual(converted["conversations"][1]["value"], "翻译\n\n学而时习之")

        raw["_has_box"] = True
        converted, reason = prepare.convert_record(raw, "系统提示", {"c2m"})
        self.assertIsNone(converted)
        self.assertEqual(reason, "has_box")

    def test_convert_record_filters_low_quality_punctuation(self):
        text = "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏"
        raw = {
            "id": "punct-one",
            "task": "punctuate",
            "instruction": "加标点",
            "input": text,
            "output": text,
            "source": "千字文",
        }
        converted, reason = prepare.convert_record(raw, "系统提示", {"punctuate"})
        self.assertIsNone(converted)
        self.assertEqual(reason, "insufficient_punctuation")

        raw["output"] = "天地玄黄，宇宙洪荒。日月盈昃，辰宿列张；寒来暑往，秋收冬藏。"
        converted, reason = prepare.convert_record(raw, "系统提示", {"punctuate"})
        self.assertIsNotNone(converted)
        self.assertEqual(reason, "accepted")

    def test_convert_record_filters_punctuation_boundary_error(self):
        raw = {
            "id": "punct-two",
            "task": "punctuate",
            "instruction": "加标点",
            "input": "学而时习之不亦说乎有朋自远方来不亦乐乎",
            "output": "。学而时习之，不亦说乎？有朋自远方来，不亦乐乎？",
            "source": "论语",
        }
        converted, reason = prepare.convert_record(raw, "系统提示", {"punctuate"})
        self.assertIsNone(converted)
        self.assertEqual(reason, "leading_boundary_punctuation")

    def test_punctuation_quote_validator_checks_order_and_nesting(self):
        self.assertTrue(prepare.quotes_are_balanced("王曰：“其言‘可也’，善。”"))
        self.assertFalse(prepare.quotes_are_balanced("王曰：“其言可也。"))
        self.assertFalse(prepare.quotes_are_balanced('其言可也。”王曰：“善。”'))

    def test_reservoir_sampling_is_bounded_and_deterministic(self):
        records = [
            {
                "id": str(index),
                "task": "c2m",
                "instruction": "翻译",
                "input": f"古文示例文本{index}",
                "output": f"现代文示例{index}",
                "source": f"来源{index}",
            }
            for index in range(20)
        ]
        first = prepare.reservoir_sample(records, {"c2m": 5}, "系统", 42, 4, 500, False, Counter())
        second = prepare.reservoir_sample(records, {"c2m": 5}, "系统", 42, 4, 500, False, Counter())
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertEqual(len(first), 5)

    def test_split_keeps_sources_isolated(self):
        records = [
            {"id": f"{source}-{index}", "source": source}
            for source in ("经", "史", "子", "集", "诗")
            for index in range(2)
        ]
        splits = prepare.split_by_source(records, seed=42, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2)
        source_sets = [{item["source"] for item in splits[name]} for name in ("train", "validation", "test")]
        self.assertTrue(all(source_sets))
        self.assertFalse(source_sets[0] & source_sets[1])
        self.assertFalse(source_sets[0] & source_sets[2])
        self.assertFalse(source_sets[1] & source_sets[2])

    def test_split_guarantees_each_task_in_eval_splits(self):
        records = []
        for task, prefix, source_count in (("c2m", "古译今", 10), ("m2c", "今译古", 10), ("punctuate", "断句", 7)):
            for source_index in range(source_count):
                for row_index in range(2):
                    records.append({
                        "id": f"{task}-{source_index}-{row_index}",
                        "task": task,
                        "source": f"{prefix}{source_index}",
                    })
        splits = prepare.split_by_source(
            records,
            seed=42,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
            min_task_sources_per_eval_split=2,
        )
        for split in ("validation", "test"):
            for task in ("c2m", "m2c", "punctuate"):
                sources = {item["source"] for item in splits[split] if item["task"] == task}
                self.assertGreaterEqual(len(sources), 2)
        source_sets = [{item["source"] for item in splits[name]} for name in ("train", "validation", "test")]
        self.assertFalse(source_sets[0] & source_sets[1])
        self.assertFalse(source_sets[0] & source_sets[2])
        self.assertFalse(source_sets[1] & source_sets[2])

    def test_split_supports_per_task_source_minimum_override(self):
        records = []
        for task, source_count in (("c2m", 8), ("punctuate", 14)):
            for source_index in range(source_count):
                records.append({
                    "id": f"{task}-{source_index}",
                    "task": task,
                    "source": f"{task}-source-{source_index}",
                })
        splits = prepare.split_by_source(
            records,
            seed=42,
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
            min_task_sources_per_eval_split=1,
            task_min_sources_per_eval_split={"punctuate": 4},
        )
        for split in ("validation", "test"):
            c2m_sources = {item["source"] for item in splits[split] if item["task"] == "c2m"}
            punct_sources = {item["source"] for item in splits[split] if item["task"] == "punctuate"}
            self.assertGreaterEqual(len(c2m_sources), 1)
            self.assertGreaterEqual(len(punct_sources), 4)


class EvaluationTests(unittest.TestCase):
    def test_character_metrics(self):
        precision, recall, f1 = evaluate.character_prf("学而 时习之", "学而时习之")
        self.assertEqual((precision, recall, f1), (1.0, 1.0, 1.0))
        self.assertEqual(evaluate.normalize_text("甲\n乙"), "甲乙")


class AuditTests(unittest.TestCase):
    def test_audit_flags_obvious_mismatch(self):
        flags, _ = audit.audit_pair(
            "元月十一日议定礼乐，十六日任命周法尚。",
            "家无完堵，地罕包桑，恒为流寓之人。",
        )
        self.assertIn("number_or_date_mismatch", flags)
        self.assertIn("low_character_overlap", flags)

    def test_audit_flags_identical_pair(self):
        flags, _ = audit.audit_pair("学而时习之，不亦说乎？", "学而时习之，不亦说乎？")
        self.assertIn("identical_input_output", flags)

    def test_punctuation_pair_is_not_identical_error(self):
        flags, _ = audit.audit_pair("学而时习之不亦说乎", "学而时习之，不亦说乎？", task="punctuate")
        self.assertNotIn("identical_input_output", flags)

    def test_punctuation_audit_flags_effectively_unpunctuated_output(self):
        text = "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏"
        flags, metrics = audit.audit_pair(text, text, task="punctuate")
        self.assertIn("insufficient_punctuation", flags)
        self.assertEqual(metrics["punctuation_count"], 0.0)

    def test_punctuation_audit_flags_text_mismatch(self):
        flags, _ = audit.audit_pair("学而时习之", "学而时习他。", task="punctuate")
        self.assertIn("punctuation_text_mismatch", flags)

    def test_punctuation_audit_flags_unjustified_leading_stop(self):
        flags, _ = audit.audit_pair(
            "学而时习之不亦说乎有朋自远方来不亦乐乎",
            "。学而时习之，不亦说乎？有朋自远方来，不亦乐乎？",
            task="punctuate",
        )
        self.assertIn("leading_boundary_punctuation", flags)

    def test_punctuation_audit_allows_stop_before_closing_quote(self):
        flags, _ = audit.audit_pair(
            "”学而时习之不亦说乎有朋自远方来不亦乐乎",
            "。”学而时习之，不亦说乎？有朋自远方来，不亦乐乎？",
            task="punctuate",
        )
        self.assertNotIn("leading_boundary_punctuation", flags)

    def test_punctuation_audit_flags_unbalanced_quotes(self):
        flags, _ = audit.audit_pair(
            "王曰学而时习之不亦说乎有朋自远方来不亦乐乎",
            "王曰：“学而时习之，不亦说乎？有朋自远方来，不亦乐乎？",
            task="punctuate",
        )
        self.assertIn("unbalanced_quotes", flags)

    def test_punctuation_audit_requires_sentence_boundary(self):
        text = "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏"
        flags, _ = audit.audit_pair(text, "天地玄黄，宇宙洪荒，日月盈昃，辰宿列张，寒来暑往，秋收冬藏", task="punctuate")
        self.assertIn("missing_sentence_boundary_punctuation", flags)


class AuditFilterTests(unittest.TestCase):
    def test_filter_removes_only_selected_flags_and_protects_gold_ids(self):
        records = [
            {"id": "safe", "task": "c2m", "source": "经"},
            {"id": "remove", "task": "m2c", "source": "史"},
            {"id": "review", "task": "c2m", "source": "子"},
        ]
        flag_map = {
            "remove": {"identical_input_output"},
            "review": {"number_or_date_mismatch"},
        }
        kept, removed, counts = audit_filter.filter_records(
            records,
            flag_map,
            {"identical_input_output", "extreme_length_ratio"},
        )
        self.assertEqual([record["id"] for record in kept], ["safe", "review"])
        self.assertEqual([record["id"] for record in removed], ["remove"])
        self.assertEqual(counts, {"identical_input_output": 1})

        with self.assertRaises(ValueError):
            audit_filter.filter_records(
                records,
                flag_map,
                {"identical_input_output"},
                protected_ids={"remove"},
            )


class AuditCandidateTests(unittest.TestCase):
    def test_select_candidates_is_task_and_flag_balanced(self):
        records = []
        for task in ("c2m", "m2c"):
            for flag in ("number_or_date_mismatch", "low_character_overlap"):
                for index in range(4):
                    records.append({
                        "id": f"{task}-{flag}-{index}",
                        "task": task,
                        "source": f"{task}-source-{index}",
                        "audit_flags": [flag],
                        "input": "输入",
                        "output": "输出",
                    })
        selected, report = audit_candidates.select_candidates(
            records,
            tasks=("c2m", "m2c"),
            flags=("number_or_date_mismatch", "low_character_overlap"),
            per_task_flag=3,
            seed=42,
        )
        self.assertEqual(len(selected), 12)
        self.assertEqual(report["tasks"], {"c2m": 6, "m2c": 6})
        self.assertEqual(len({record["id"] for record in selected}), 12)

    def test_select_candidates_can_keep_a_short_bucket(self):
        records = [
            {
                "id": "only-one",
                "task": "c2m",
                "source": "经",
                "audit_flags": ["number_or_date_mismatch"],
                "input": "输入",
                "output": "输出",
            }
        ]
        selected, report = audit_candidates.select_candidates(
            records,
            tasks=("c2m",),
            flags=("number_or_date_mismatch",),
            per_task_flag=5,
            seed=42,
            allow_short_buckets=True,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(report["buckets"]["c2m"]["number_or_date_mismatch"], 1)


class AuditReviewSummaryTests(unittest.TestCase):
    def test_summary_counts_review_statuses_by_audit_flag(self):
        candidates = [
            {"id": "one", "task": "c2m", "audit_flags": ["number_or_date_mismatch"]},
            {"id": "two", "task": "m2c", "audit_flags": ["low_character_overlap"]},
        ]
        reviews = [
            {
                "id": "one", "task": "c2m", "audit_flags": ["number_or_date_mismatch"],
                "review_status": "approved", "approved_reference": "", "review_notes": "可用",
                "reviewer": "gpt-5.6-sol", "confidence": "high",
            },
            {
                "id": "two", "task": "m2c", "audit_flags": ["low_character_overlap"],
                "review_status": "rejected", "approved_reference": "", "review_notes": "错配",
                "reviewer": "gpt-5.6-sol", "confidence": "high",
            },
        ]
        merged, report = audit_summary.summarize_reviews(candidates, reviews)
        self.assertEqual(len(merged), 2)
        self.assertEqual(report["statuses"], {"approved": 1, "rejected": 1})
        self.assertEqual(report["flags"]["number_or_date_mismatch"]["approved"], 1)
        self.assertEqual(report["flags"]["low_character_overlap"]["rejected"], 1)


class GoldCandidateTests(unittest.TestCase):
    def test_selection_is_balanced_and_excludes_flagged_ids(self):
        records = []
        for task in ("c2m", "m2c", "punctuate"):
            for index in range(6):
                records.append({
                    "id": f"{task}-{index}",
                    "task": task,
                    "source": f"{task}-source-{index % 3}",
                    "conversations": [
                        {"from": "human", "value": f"指令\n\n输入{index}"},
                        {"from": "gpt", "value": f"输出{index}"},
                    ],
                })
        selected = gold.select_candidates(records, per_task=4, seed=42, excluded_ids={"c2m-0"})
        counts = Counter(item["task"] for item in selected)
        self.assertEqual(counts, {"c2m": 4, "m2c": 4, "punctuate": 4})
        self.assertNotIn("c2m-0", {item["id"] for item in selected})


class MergeReviewTests(unittest.TestCase):
    def test_merge_uses_corrected_reference_and_filters_rejected(self):
        candidates = [
            {"id": "one", "task": "c2m", "prompt": "甲", "reference": "旧答案", "source": "经"},
            {"id": "two", "task": "m2c", "prompt": "乙", "reference": "答案", "source": "史"},
        ]
        reviews = [
            {"id": "one", "task": "c2m", "review_status": "approved", "approved_reference": "新答案",
             "review_notes": "修正", "reviewer": "gpt-5.6-sol", "confidence": "high"},
            {"id": "two", "task": "m2c", "review_status": "rejected", "approved_reference": "",
             "review_notes": "错配", "reviewer": "gpt-5.6-sol", "confidence": "high"},
        ]
        merged, approved, report = merge_reviews.merge_reviews(candidates, reviews)
        self.assertEqual(len(merged), 2)
        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0]["reference"], "新答案")
        self.assertEqual(approved[0]["reviewer"], "gpt-5.6-sol")
        self.assertEqual(approved[0]["review_confidence"], "high")
        self.assertEqual(report["statuses"], {"approved": 1, "rejected": 1})
        self.assertEqual(report["approved_high_confidence_records"], 1)


class ApplyAuditReviewTests(unittest.TestCase):
    def test_apply_removes_rejected_and_replaces_corrected_reference(self):
        from tools import apply_gujinbridge_audit_reviews as apply_reviews

        records = [
            {"id": "keep", "task": "c2m", "source": "甲", "conversations": [
                {"from": "human", "value": "题目"}, {"from": "gpt", "value": "旧答案"},
            ]},
            {"id": "drop", "task": "m2c", "source": "乙", "conversations": [
                {"from": "human", "value": "题目"}, {"from": "gpt", "value": "坏答案"},
            ]},
        ]
        reviews = {
            "keep": {"id": "keep", "review_status": "approved", "reference": "旧答案",
                     "approved_reference": "新答案", "audit_flags": ["low_character_overlap"]},
            "drop": {"id": "drop", "review_status": "rejected", "reference": "坏答案",
                     "approved_reference": "", "audit_flags": ["low_character_overlap"]},
        }
        kept, removed, corrected, found = apply_reviews.apply_reviews(records, reviews, "train")
        self.assertEqual([record["id"] for record in kept], ["keep"])
        self.assertEqual(kept[0]["conversations"][1]["value"], "新答案")
        self.assertEqual([record["id"] for record in removed], ["drop"])
        self.assertEqual(corrected[0]["original_reference"], "旧答案")
        self.assertEqual(found, {"keep", "drop"})


class FinalizeV5Tests(unittest.TestCase):
    def test_finalize_removes_only_selected_risks(self):
        from tools import finalize_gujinbridge_v5 as finalize

        records = [
            {"id": "clean", "task": "c2m"},
            {"id": "risk", "task": "m2c"},
            {"id": "approved-risk", "task": "m2c"},
        ]
        kept, removed = finalize.finalize_records(records, {"risk"})
        self.assertEqual([record["id"] for record in kept], ["clean", "approved-risk"])
        self.assertEqual([record["id"] for record in removed], ["risk"])


if __name__ == "__main__":
    unittest.main()
