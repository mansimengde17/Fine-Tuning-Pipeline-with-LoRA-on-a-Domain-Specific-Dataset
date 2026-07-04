import json
import os
import sys
import unittest

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from data.build_dataset import build
from finetune.evaluate import base_model, finetuned_model, run_benchmark
from finetune.train import expand_sweep, load_config, train_simulated


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = build()

    def test_split_sizes(self):
        self.assertGreater(self.counts["train"], self.counts["val"])
        self.assertGreater(self.counts["train"], self.counts["test"])

    def test_no_template_leakage(self):
        train_templates = {r["template_id"]
                           for r in load_jsonl("data/train.jsonl")}
        test_templates = {r["template_id"]
                          for r in load_jsonl("data/test.jsonl")}
        self.assertFalse(train_templates & test_templates)


class TrainingTests(unittest.TestCase):
    def test_simulated_run_reproducible_and_checkpointed(self):
        build()
        config = load_config("configs/lora_r16.yaml")
        a = train_simulated(config)
        b = train_simulated(config)
        self.assertEqual(a["best_val_loss"], b["best_val_loss"])
        self.assertTrue(os.path.exists(
            os.path.join(a["adapter"], "adapter_config.json")))

    def test_sweep_expansion(self):
        configs = expand_sweep("configs/sweep.yaml")
        self.assertEqual(len(configs), 27)
        self.assertEqual(len({c["run_name"] for c in configs}), 27)


class EvalTests(unittest.TestCase):
    def test_finetuned_beats_base_on_benchmark(self):
        build()
        cases = load_jsonl("data/benchmark.jsonl") + \
            load_jsonl("data/test.jsonl")
        base = run_benchmark(base_model(), cases)
        tuned = run_benchmark(finetuned_model(), cases)
        self.assertGreater(tuned["accuracy"], base["accuracy"])


if __name__ == "__main__":
    unittest.main()
