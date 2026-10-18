"""CPU-only tests of adapter forensics; generated fixtures are not checkpoints."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from .hf_backend import MODEL_ID, MODEL_REVISION
from .inspect_adapter import inspect_checkpoint


class AdapterForensicsTests(unittest.TestCase):
    def fixture(self, root, b, updates=1):
        target = "model.language_model.layers.40.self_attn.q_proj"
        a = torch.tensor([[1., 2., 3.], [0., 1., 1.]])
        save_file({f"base_model.model.{target}.lora_A.weight": a,
                   f"base_model.model.{target}.lora_B.weight": b}, str(root / "adapter_model.safetensors"))
        (root / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": MODEL_ID, "revision": MODEL_REVISION,
            "r": 2, "lora_alpha": 4, "target_modules": [target], "init_lora_weights": True,
        }))
        (root / "trainer_state.json").write_text(json.dumps({"optimizer_updates": updates, "seed": 7301}))
        return a

    def test_nonzero_effective_delta_agrees_with_explicit_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            b = torch.tensor([[0.2, 0.1], [0.0, 0.3], [0.4, 0.0], [0.1, 0.1]])
            a = self.fixture(root, b)
            report = inspect_checkpoint(root)
            self.assertTrue(report["evidence_of_nonzero_update"])
            self.assertAlmostEqual(report["modules"][0]["effective_delta_frobenius_norm"],
                                   float(torch.linalg.vector_norm(2 * b.double() @ a.double())), places=10)
            self.assertEqual(report["initialization"]["scope"], "not_attested_in_checkpoint_metadata")
            self.assertFalse(report["accuracy_or_quality_inferred"])

    def test_zero_or_nonfinite_adapter_is_not_evidence_of_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root, torch.zeros((4, 2)))
            report = inspect_checkpoint(root)
            self.assertFalse(report["evidence_of_nonzero_update"])
            self.assertIn("successful_updates_claimed_but_no_effective_adapter_delta", report["problems"])
            self.fixture(root, torch.full((4, 2), float("nan")))
            report = inspect_checkpoint(root)
            self.assertFalse(report["finite"])
            self.assertFalse(report["evidence_of_nonzero_update"])


if __name__ == "__main__":
    unittest.main()
