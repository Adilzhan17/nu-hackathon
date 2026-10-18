"""Small local Gemma 4 QLoRA backend; no network calls, subprocesses or GPU locks.

The caller owns the shared GPU lock, time budget, split audit and model download.
Use the exact MODEL_ID/REVISION below. This NF4 backend is NOT the deployed
llama.cpp QAT Q4_0 runtime; evaluate adapter disabled/enabled on this same base.

API: load_backend(path, training=False, adapter_path=None) -> Backend.
Backend.encode([PIL images], prompt, target=None) -> encoded batch; train_batch
performs one optimizer update, predict generates text, save_adapter saves only
LoRA and processor files. No prompt text or frames leave this machine.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
import re
import time
from typing import Any

MODEL_ID = "google/gemma-4-12B-it-qat-q4_0-unquantized"
MODEL_REVISION = "b6ed86275a6a5735884e208bfed95b445a684ca2"
REVISION = MODEL_REVISION
WEIGHTS_SHA256 = "26f2cee4292298a3f9f92209643c37c80e34e011381e22434088870d9439a0a0"
WEIGHTS_BYTES = 23_919_549_408
DEPENDENCY_PINS = {
    "torch": "2.8.0", "torchvision": "0.23.0", "transformers": "5.10.1",
    "peft": "0.19.0", "bitsandbytes": "0.48.2", "accelerate": "1.10.1",
}
SKIP_QUANTIZATION = ["lm_head", "model.embed_vision", "model.embed_audio"]
MAX_SOFT_TOKENS = 70


def keep_vision_in_fp32(vision):
    """Run the small visual embedder in FP32, without upcasting token embeddings.

    Observed on the real T4 pilot: FP16 patch_dense overflowed despite finite
    inputs/weights (277 nonfinite outputs), before its following normalization.
    Disabling AMP alone is insufficient while its parameters remain FP16.
    """
    import torch
    if getattr(vision, "_gemma_fp32_vision", False):
        return
    vision.float()
    original_forward = vision.forward

    def forward(*args, **kwargs):
        # The architecture itself casts raw patches to patch_dense.weight.dtype;
        # all visual LayerNorms and the two small projections therefore stay FP32.
        with torch.autocast(device_type=next(vision.parameters()).device.type, enabled=False):
            return original_forward(*args, **kwargs)

    vision.forward = forward
    vision._gemma_fp32_vision = True


def adapter_target_names(names: list[str], num_layers: int = 48) -> list[str]:
    """Limit updates to q/v projections in the final eight language layers."""
    expression = re.compile(r"model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)$")
    return [name for name in names if (match := expression.fullmatch(name))
            and num_layers - 8 <= int(match.group(1)) < num_layers]


def _versions() -> dict[str, str]:
    result = {}
    for package in DEPENDENCY_PINS:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not_installed"
    return result


def answer_only_loss(logits, target_ids):
    """Align last N+1 logits to N assistant tokens, excluding the final logit."""
    import torch
    if logits.shape[1] != target_ids.shape[1] + 1:
        raise ValueError("Answer/logit alignment failed")
    prediction = logits[:, :-1, :]
    return torch.nn.functional.cross_entropy(
        prediction.float().reshape(-1, prediction.shape[-1]), target_ids.reshape(-1),
    )


class Backend:
    def __init__(self, model, processor, *, training: bool, model_path: Path,
                 adapter_path: Path | None, targets: list[str]):
        import torch
        self.torch = torch
        self.model = model
        self.processor = processor
        self.training = training
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.targets = targets
        self.device = torch.device("cuda:0")
        self.updates = 0
        self.examples_used = 0
        self.started_at = time.time()
        self.model.config.use_cache = False if training else True
        self.model.config.get_text_config().use_cache = False if training else True

    def encode(self, frames: list, prompt: str, target: str | None = None,
               *, max_tokens: int = 1024, max_target_tokens: int = 192) -> dict[str, Any]:
        """Encode 2–4 ordered frames; supervise ONLY the assistant continuation.

        Returns CPU tensors. Image tokens, prompt tokens and padding never enter
        the objective. Limits reject examples; they never truncate image tokens
        or silently truncate an answer. The prompt must contain the timestamps.
        """
        torch = self.torch
        if not 2 <= len(frames) <= 4:
            raise ValueError("Expected 2–4 ordered frames per example")
        images = [frame.convert("RGB") for frame in frames]
        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image"} for _ in images)
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = dict(self.processor(
            text=[text], images=[images], return_tensors="pt", padding=False,
            images_kwargs={"max_soft_tokens": MAX_SOFT_TOKENS},
        ))
        prompt_tokens = int(inputs["input_ids"].shape[1])
        target_ids = None
        if target is not None:
            if not target.strip():
                raise ValueError("An empty assistant target is not training data")
            target_ids = self.processor.tokenizer(
                target + "<turn|>", add_special_tokens=False, return_tensors="pt",
            )["input_ids"]
            count = int(target_ids.shape[1])
            if count > max_target_tokens:
                raise ValueError(f"Assistant target has {count} tokens; cap {max_target_tokens}")
            inputs["input_ids"] = torch.cat((inputs["input_ids"], target_ids), dim=1)
            # The processor expands image placeholders and supplies multimodal
            # token types. Assistant text adds ordinary text positions only.
            for name in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
                if name in inputs:
                    fill = 1 if name == "attention_mask" else 0
                    extension = torch.full((1, count), fill, dtype=inputs[name].dtype)
                    inputs[name] = torch.cat((inputs[name], extension), dim=1)
        total_tokens = int(inputs["input_ids"].shape[1])
        if total_tokens > max_tokens:
            raise ValueError(f"Encoded example has {total_tokens} tokens; cap {max_tokens}")
        return {"inputs": inputs, "target_ids": target_ids,
                "prompt_tokens": prompt_tokens, "total_tokens": total_tokens,
                "image_count": len(images), "max_soft_tokens_per_image": MAX_SOFT_TOKENS}

    def _device_inputs(self, batch):
        # Position IDs and image masks must retain integer/bool dtypes.
        return {key: value.to(self.device) if self.torch.is_tensor(value) else value
                for key, value in batch["inputs"].items()}

    def make_optimizer(self, learning_rate: float = 1e-4):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not self.training or not params:
            raise ValueError("Backend has no trainable adapter")
        return self.torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)

    def make_scaler(self):
        return self.torch.amp.GradScaler("cuda", init_scale=256.0)

    def train_batch(self, batch: dict, optimizer, scaler=None,
                    *, loss_weight: float = 1.0) -> dict[str, Any]:
        """One finite-gradient, answer-only update. Nonfinite updates are skipped.

        Read `updated` before counting an example as actually used. A skipped
        update remains an error for a short feasibility pilot, not success.
        """
        if not self.training or batch.get("target_ids") is None:
            raise ValueError("Training backend and assistant target are required")
        if not isinstance(loss_weight, (int, float)) or not math.isfinite(loss_weight) or not 0 < loss_weight <= 1:
            raise ValueError("loss_weight must be finite and in (0, 1]")
        torch = self.torch
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        inputs = self._device_inputs(batch)
        target = batch["target_ids"].to(self.device)
        target_length = int(target.shape[1])
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.float16):
            # Keep the position immediately before the answer plus its positions.
            # Do not allocate a vocab262144 logit matrix for every image/prompt
            # position. Gemma4Unified's verified API supports logits_to_keep.
            output = self.model(**inputs, use_cache=False, logits_to_keep=target_length + 1)
            loss = answer_only_loss(output.logits, target)
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            optimizer.zero_grad(set_to_none=True)
            return {"loss": None, "updated": False, "finite": False,
                    "reason": "nonfinite_loss", "elapsed_sec": time.monotonic() - started}
        weighted_loss = loss * loss_weight
        if scaler is not None:
            scaler.scale(weighted_loss).backward()
            scaler.unscale_(optimizer)
        else:
            weighted_loss.backward()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        grads = [p.grad for p in trainable if p.grad is not None]
        if not grads:
            raise RuntimeError("No LoRA gradients: no actual training occurred")
        finite = all(bool(torch.isfinite(grad).all()) for grad in grads)
        if not finite:
            if scaler is not None:
                # GradScaler has recorded the overflow and skips optimizer.step.
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            return {"loss": loss_value, "updated": False, "finite": False,
                    "reason": "nonfinite_gradients", "elapsed_sec": time.monotonic() - started}
        norm = torch.nn.utils.clip_grad_norm_(trainable, 0.3, error_if_nonfinite=True)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        self.updates += 1
        self.examples_used += int(target.shape[0])
        optimizer.zero_grad(set_to_none=True)
        return {"loss": loss_value, "weighted_loss": loss_value * loss_weight,
                "loss_weight": loss_weight, "grad_norm": float(norm.detach().cpu()),
                "updated": True, "finite": True, "updates": self.updates,
                "examples_used": self.examples_used, "answer_tokens": target_length,
                "input_tokens": batch["total_tokens"],
                "elapsed_sec": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()}

    def predict(self, frames: list, prompt: str, *, max_new_tokens: int = 96,
                adapter_enabled: bool = True) -> str:
        """Greedy local decoding; disable adapter for the paired HF baseline."""
        from contextlib import nullcontext
        torch = self.torch
        batch = self.encode(frames, prompt)
        inputs = self._device_inputs(batch)
        self.model.eval()
        context = self.model.disable_adapter() if not adapter_enabled and hasattr(self.model, "disable_adapter") else nullcontext()
        stop_ids = list({self.processor.tokenizer.eos_token_id,
                         self.processor.tokenizer.convert_tokens_to_ids("<turn|>")})
        with context, torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, eos_token_id=stop_ids,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                disable_compile=True,
            )
        continuation = generated[0, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(continuation, skip_special_tokens=True)

    def provenance(self) -> dict[str, Any]:
        return {
            "base_model_id": MODEL_ID, "base_model_revision": MODEL_REVISION,
            "expected_base_weights_sha256": WEIGHTS_SHA256,
            "weight_checksum_verified_by_backend": False,
            "base_model_path": str(self.model_path), "license": "Apache-2.0",
            "architecture": "Gemma4UnifiedForConditionalGeneration",
            "quantization": "bitsandbytes NF4 double quantization; FP16 compute",
            "visual_embedder_compute_dtype": "float32 with autocast disabled",
            "visual_precision_reason": "Real T4 FP16 patch_dense overflow before normalization; frozen visual weights only promoted",
            "quantization_exclusions": SKIP_QUANTIZATION,
            "deployed_gguf_equivalent": False,
            "comparison_requirement": "Same HF NF4 base, frames, prompt and decoding; adapter disabled versus enabled",
            "lora_rank": 4, "lora_alpha": 8, "target_modules": self.targets,
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "versions": _versions(), "optimizer_updates": self.updates,
            "examples_used": self.examples_used, "max_soft_tokens_per_image": MAX_SOFT_TOKENS,
            "audio_used": False, "external_ai_requests": False,
        }

    def save_adapter(self, output_dir: str | Path, extra_provenance: dict | None = None) -> dict:
        """Save a real adapter only after at least one successful update."""
        if not self.training or self.updates < 1 or not hasattr(self.model, "peft_config"):
            raise ValueError("Cannot save an untrained or non-adapter checkpoint as fine-tuning")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for config in self.model.peft_config.values():
            config.base_model_name_or_path = MODEL_ID
            config.revision = MODEL_REVISION
        self.model.save_pretrained(output, safe_serialization=True, save_embedding_layers=False)
        self.processor.save_pretrained(output)
        provenance = self.provenance()
        if extra_provenance:
            provenance["experiment"] = extra_provenance
        (output / "training_provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return provenance


def load_backend(model_path: str | Path, training: bool = False,
                 adapter_path: str | Path | None = None) -> Backend:
    """Load local pinned weights; no download or other environment mutation.

    Caller must verify MODEL_REVISION/WEIGHTS_SHA256 when staging the model.
    Local files only prevents token lookup, gated access, or hidden network use.
    """
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model

    path = Path(model_path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "gemma4_unified":
        raise ValueError("Expected exact Gemma 4 Unified training base")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this bounded NF4 pilot")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_storage=torch.uint8,
        llm_int8_skip_modules=SKIP_QUANTIZATION,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        quantization_config=quantization, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    # The vision path explicitly casts pixels to patch_dense.weight.dtype. A
    # quantized uint8 patch_dense would corrupt pixels before LayerNorm.
    patch_dense = model.model.embed_vision.patch_dense
    if patch_dense.weight.dtype != torch.float16:
        raise RuntimeError("Vision embedder must remain FP16, not quantized")
    if model.get_input_embeddings().weight.dtype != torch.float16:
        raise RuntimeError("Frozen shared token embeddings must remain FP16")
    keep_vision_in_fp32(model.model.embed_vision)
    if model.model.embed_vision.patch_dense.weight.dtype != torch.float32:
        raise RuntimeError("Visual projection stabilization was not applied")
    model.requires_grad_(False)
    targets = adapter_target_names([name for name, _ in model.named_modules()],
                                   config["text_config"]["num_hidden_layers"])
    if len(targets) != 14:
        raise RuntimeError(f"Expected 14 final-eight-layer q/v modules, found {len(targets)}")
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model, str(Path(adapter_path).resolve()), is_trainable=training,
            local_files_only=True,
        )
    elif training:
        model = get_peft_model(model, LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=targets,
        ))
    if training:
        # Unlike prepare_model_for_kbit_training, this does not promote the
        # billion-parameter embedding to FP32 or train/save its weights.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in name:
                    raise RuntimeError("Unexpected trainable non-LoRA parameter: " + name)
                parameter.data = parameter.data.to(torch.float32)
        model.train()
    else:
        model.eval()
    return Backend(model, processor, training=training, model_path=path,
                   adapter_path=Path(adapter_path).resolve() if adapter_path else None, targets=targets)
