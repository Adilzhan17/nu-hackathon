"""CPU checks for supervision alignment; no models, network or GPU required."""
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ml.gemma_lab.adaptation.hf_backend import Backend, adapter_target_names, answer_only_loss, keep_vision_in_fp32


class _Image:
    def convert(self, mode):
        assert mode == "RGB"
        return self


class _Tokenizer:
    def __call__(self, text, **kwargs):
        assert text.endswith("<turn|>")
        assert kwargs["add_special_tokens"] is False
        return {"input_ids": torch.tensor([[40, 41, 106]])}


class _Processor:
    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        assert len(messages[0]["content"]) == 3
        return "prompt-with-image-placeholders"

    def __call__(self, **kwargs):
        assert kwargs["images_kwargs"] == {"max_soft_tokens": 70}
        return {"input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]]),
                "image_position_ids": torch.tensor([[[0, 0], [1, 0]]]),
                "pixel_values": torch.ones((2, 2, 3))}


class BackendContractTests(unittest.TestCase):
    def backend(self):
        backend = Backend.__new__(Backend)
        backend.torch = torch
        backend.processor = _Processor()
        return backend

    def test_only_final_eight_language_layers_targeted(self):
        names = [f"model.language_model.layers.{i}.self_attn.{projection}"
                 for i in range(48) for projection in ("q_proj", "v_proj")
                 if projection == "q_proj" or i % 6 != 5]
        names += ["model.embed_vision.q_proj", "lm_head", "model.embed_audio.v_proj"]
        selected = adapter_target_names(names)
        self.assertEqual(len(selected), 14)
        self.assertTrue(all(40 <= int(name.split(".")[3]) <= 47 for name in selected))

    def test_image_prompt_positions_never_become_targets(self):
        encoded = self.backend().encode([_Image(), _Image()], "t=0, t=1", '{"action":"leave"}')
        self.assertEqual(encoded["inputs"]["input_ids"].tolist(), [[1, 2, 3, 4, 40, 41, 106]])
        self.assertEqual(encoded["target_ids"].tolist(), [[40, 41, 106]])
        self.assertEqual(encoded["inputs"]["mm_token_type_ids"].tolist(), [[0, 1, 1, 0, 0, 0, 0]])
        self.assertEqual(encoded["inputs"]["image_position_ids"].dtype, torch.long)

    def test_oversized_examples_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "Encoded example"):
            self.backend().encode([_Image(), _Image()], "x", "answer", max_tokens=6)
        with self.assertRaisesRegex(ValueError, "Assistant target"):
            self.backend().encode([_Image(), _Image()], "x", "answer", max_target_tokens=2)

    def test_answer_loss_uses_preceding_position_and_excludes_last(self):
        logits = torch.tensor([[[0., 8., 0.], [0., 0., 8.], [100., -100., 0.]]], requires_grad=True)
        target = torch.tensor([[1, 2]])
        loss = answer_only_loss(logits, target)
        self.assertLess(float(loss), 0.001)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[:, -1, :], torch.zeros((1, 3))))
        self.assertGreater(float(logits.grad[:, :-1, :].abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "alignment"):
            answer_only_loss(logits[:, :-1, :], target)

    def test_weak_weight_changes_actual_gradient_update(self):
        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([[[0., 5., 0.], [0., 0., 5.], [0., 0., 0.]]]))

            def forward(self, **kwargs):
                return SimpleNamespace(logits=self.logits)

        changes = []
        for weight in (1.0, 0.2):
            backend = self.backend()
            backend.model, backend.training = SmallModel(), True
            backend.device, backend.updates, backend.examples_used = torch.device("cpu"), 0, 0
            before = backend.model.logits.detach().clone()
            optimizer = torch.optim.SGD(backend.model.parameters(), lr=0.1)
            with patch("torch.autocast", return_value=nullcontext()), patch("torch.cuda.max_memory_allocated", return_value=0):
                result = backend.train_batch({"inputs": {}, "target_ids": torch.tensor([[1, 2]]),
                                              "total_tokens": 10}, optimizer, loss_weight=weight)
            self.assertTrue(result["updated"])
            changes.append((backend.model.logits.detach() - before).abs().sum().item())
        self.assertAlmostEqual(changes[1] / changes[0], 0.2, places=3)

    def test_visual_projection_overflow_is_prevented_before_normalization(self):
        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_dense = torch.nn.Linear(4, 4, bias=False)
                self.norm = torch.nn.LayerNorm(4)
                with torch.no_grad():
                    self.patch_dense.weight.copy_(torch.tensor([[150., 150., 150., 150.],
                                                               [100., 100., 100., 100.],
                                                               [80., 80., 80., 80.],
                                                               [-150., -150., -150., -150.]]))

            def forward(self, pixels):
                return self.norm(self.patch_dense(pixels.to(self.patch_dense.weight.dtype)))

        vision = Vision().half()
        pixels = torch.full((1, 4), 150.)
        self.assertFalse(bool(torch.isfinite(vision(pixels)).all()))
        unrelated_large_embedding = torch.nn.Embedding(10, 4).half()
        keep_vision_in_fp32(vision)
        self.assertTrue(bool(torch.isfinite(vision(pixels)).all()))
        self.assertEqual(vision.patch_dense.weight.dtype, torch.float32)
        self.assertEqual(unrelated_large_embedding.weight.dtype, torch.float16)
        keep_vision_in_fp32(vision)
        self.assertTrue(bool(torch.isfinite(vision(pixels)).all()))


if __name__ == "__main__":
    unittest.main()
