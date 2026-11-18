import json
import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel

from rgcl.analysis import classification_records, line_token_mass, summarize_classification
from rgcl.config import REGION_METHODS, check_run
from rgcl.data import make_feature, modified_lines
from rgcl.metrics import paired_metrics
from rgcl.model import Detector, load_checkpoint, save_checkpoint
from rgcl.training import (
    AdamW, load_training_state, randomize_regions, region_supervision_statistics,
    restrict_region_supervision, restore_rng, save_training_state,
)


class CoreChecks(unittest.TestCase):
    @staticmethod
    def detector(method):
        config = RobertaConfig(vocab_size=24, hidden_size=16, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=32)
        config._attn_implementation = "eager"
        return Detector(RobertaModel(config, add_pooling_layer=False), config, method, 1.)

    @staticmethod
    def inputs():
        return dict(input_ids=torch.tensor([[0, 4, 5, 6, 2], [0, 7, 8, 9, 2]]),
                    attention_mask=torch.ones((2, 5), dtype=torch.long),
                    code_mask=torch.tensor([[False, True, True, True, False]] * 2),
                    line_token_mask=torch.tensor([[[False, True, True, False, False],
                                                    [False, False, True, True, False],
                                                    [False, False, False, False, False]]] * 2))

    def test_method_definitions_and_rng(self):
        methods = ("rgcl", "rgcl_no_repair_guidance", "rgcl_no_interaction")
        models, rng = {}, {}
        for method in methods:
            torch.manual_seed(1234)
            models[method] = self.detector(method).eval()
            rng[method] = torch.get_rng_state()
        self.assertIn("rgcl_no_interaction", REGION_METHODS)
        self.assertNotIn("rgcl_no_repair_guidance", REGION_METHODS)
        for method in methods[1:]:
            self.assertTrue(torch.equal(rng[method], rng["rgcl"]))
            for key, value in models[method].state_dict().items():
                self.assertTrue(torch.equal(models["rgcl"].state_dict()[key], value))
        self.assertFalse(models["rgcl_no_interaction"].line_function_interaction)
        self.assertTrue(models["rgcl_no_repair_guidance"].line_function_interaction)
        self.assertFalse(hasattr(models["rgcl_no_interaction"], "context_u"))
        batch = self.inputs()
        kwargs = dict(labels=torch.tensor([1, 0]), line_labels=torch.tensor([[1, 0, -1], [0, 1, -1]]))
        outputs = {method: model(**batch, **kwargs) for method, model in models.items()}
        torch.testing.assert_close(outputs["rgcl"]["probability"], outputs[methods[1]]["probability"], rtol=0, atol=0)
        self.assertEqual(float(outputs[methods[1]]["soft_regional_loss"].detach()), 0.)
        torch.testing.assert_close(outputs["rgcl"]["soft_regional_loss"], torch.tensor(math.log(2)))
        outputs["rgcl"]["loss"].backward()
        self.assertGreater(float(models["rgcl"].context_v.weight.grad.norm()), 0)
        with self.assertRaises(ValueError):
            check_run("codebert", "primevul_paired", 42, "rgcl_no_interaction")

    def test_effective_mass_and_pooling_weight(self):
        model = self.detector("rgcl")
        batch = self.inputs()
        states = torch.arange(2 * 5 * 16, dtype=torch.float32).reshape(2, 5, 16)
        z, out = model.bounded_pool(states, batch["code_mask"], batch["line_token_mask"])
        torch.testing.assert_close(out["effective_token_mass"], torch.tensor([[1.5, 1.5, 0.]] * 2))
        torch.testing.assert_close(out["line_emphasis_factors"], torch.tensor([[3., 3., 0.]] * 2))
        torch.testing.assert_close(out["pooling_weights"], torch.tensor([[.5, .5, 0.]] * 2))
        torch.testing.assert_close(z, states[:, 1:4].mean(1))
        foreground = torch.tensor([[5., 1., 0.], [1., 5., 0.]], requires_grad=True)
        loss, eligible = model.soft_regional_loss(foreground, out["effective_token_mass"],
                                                  torch.tensor([[0, 1, -1], [0, 0, -1]]))
        self.assertEqual(int(eligible), 1)
        self.assertAlmostEqual(float(loss.detach()), math.log(10), places=6)
        loss.backward()
        self.assertEqual(float(foreground.grad[1].abs().sum()), 0.)

    def test_dual_edit_and_native_mask(self):
        before = "int f() {\n    return 1;\n}\n"
        after = "int f() {\n    return 2;\n}\n"
        self.assertEqual(modified_lines(before, after), [1])
        self.assertEqual(modified_lines(after, before), [1])
        encoding = dict(line_numbers=[0, 1], line_complete=[True, False])
        row = dict(target=1, lines_ground_truth=[1])
        self.assertEqual(make_feature(row, None, encoding=encoding)["line_labels"], [0, 1])
        row["diff_reliable"] = True
        self.assertEqual(make_feature(row, None, encoding=encoding)["line_labels"], [0, 0])
        row["lines_ground_truth"] = [0, 3]
        self.assertEqual(make_feature(row, None, encoding=encoding)["line_labels"], [1, 0])

    def test_random_region_and_side_ablation(self):
        f = dict(line_token_indices=[[0, 1], [1, 2]], code_mask=[True] * 3, line_labels=[1, 0], labels=1)
        other = dict(f, labels=0)
        self.assertEqual(line_token_mass(f), [1.5, 1.5])
        first, second = [dict(f), dict(other)], [dict(f), dict(other)]
        self.assertEqual(randomize_regions(first, 42), randomize_regions(second, 42))
        self.assertEqual(first, second)
        for current in first:
            self.assertEqual(sum(n * bit for n, bit in zip(line_token_mass(current), current["line_labels"])), 1.5)
        sides = region_supervision_statistics(first, 0)
        self.assertEqual(sides["supervised_functions"], 1)
        restrict_region_supervision(first, 0)
        self.assertEqual(first[0]["line_labels"], [0, 0])
        self.assertEqual([r["labels"] for r in first], [1, 0])

    def test_selected_checkpoint_without_diff(self):
        for method in ("rgcl_no_repair_guidance", "plm_classifier"):
            with self.subTest(method=method), TemporaryDirectory() as tmp:
                model = self.detector(method).eval()
                batch = self.inputs()
                expected = model(**batch)["probability"]
                root = Path(tmp)
                model.config.save_pretrained(root)
                path = root / "best.safetensors"
                save_checkpoint(path, model, backbone="unixcoder", dataset="primevul_paired", seed=42,
                                method=method, class_weight=1, selected_by="p_c", epoch=1)
                restored = load_checkpoint(path, model_source=root)
                self.assertEqual(restored.method, method)
                torch.testing.assert_close(restored(**batch)["probability"], expected, rtol=0, atol=0)

    def test_resume_restores_optimizer_rng_and_history(self):
        torch.manual_seed(22)
        random.seed(22)
        np.random.seed(22)
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            log = run / "best.training.jsonl"
            log.write_text(json.dumps({"epoch": 1}) + "\n")
            model = torch.nn.Linear(3, 1)
            optimizer = AdamW(model.parameters())
            model(torch.ones((1, 3))).sum().backward()
            optimizer.step()
            save_training_state(run, model, optimizer, {"seed": 22}, {"next_batch": 1})
            expected = torch.rand(4)
            with log.open("a") as stream:
                stream.write(json.dumps({"epoch": 2}) + "\n")
            restored = torch.nn.Linear(3, 1)
            restore_optimizer = AdamW(restored.parameters())
            loop, rng = load_training_state(run, restored, restore_optimizer, {"seed": 22})
            restore_rng(rng)
            self.assertEqual(loop, {"next_batch": 1})
            torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)
            torch.testing.assert_close(restored.weight, model.weight, rtol=0, atol=0)
            self.assertEqual(log.read_text(), json.dumps({"epoch": 1}) + "\n")
            self.assertTrue(all(state["step"] == 1 for state in restore_optimizer.state.values()))

    def test_fixed_pair_threshold_and_state_transition(self):
        rows = [dict(sample_key=str(i), pair_key="pair", target=1 - i) for i in range(2)]
        no_repair, rgcl = (.51, .8), (.51, .5)
        self.assertEqual(paired_metrics(rows, list(rgcl))["pair_counts"]["p_c"], 1)
        predictions = {method: [dict(r, probability=p, prediction=int(p > .5)) for r, p in zip(rows, values)]
                       for method, values in (("rgcl_no_repair_guidance", no_repair), ("rgcl", rgcl))}
        records = classification_records(rows, predictions, 42)
        summary = summarize_classification(records)
        self.assertEqual(summary["matrix"][1][0], 1)
        self.assertEqual(summary["pc"]["net"], 1)


if __name__ == "__main__":
    unittest.main()
