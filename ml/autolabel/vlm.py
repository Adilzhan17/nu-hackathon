"""In-process inference from local weights. No remote code, cloud clients, audio or identity."""
import json
import os
from pathlib import Path
from .schema import ACTIONS, OBJECTS

PROMPT = '''Analyze the ordered timestamped frames as a short temporal sequence.
Describe ONLY visually supported actions. Do not identify people, read personal details,
analyze sound, infer intent, or decide whether conduct is legal or violates rules.
Missing visibility in a frame does NOT establish absence, negligence, or a violation.
A desk, counter or cabinet is NOT automatically a safe. Paper is NOT automatically money.
Keys and stamps are small: use unidentified_item when their identity is unclear.
Do not infer object identity solely from the setting. If unsure, say so.
Use only these action codes: {actions}.
Use only these object names: {objects}.
Actions are from the visible worker's perspective. cash_received means receiving cash
from another person; cash_transferred means handing cash to another person.
cash_counting requires a visible counting sequence; moving a bundle is not counting.
safe_opening and safe_closing require visible door motion with before/after evidence.
leave_workstation requires a visible departure sequence, not merely an empty chair.
Return at most 3 events. If no supported target action is visible, return events=[].
Each event needs at least TWO different supplied timestamps showing the sequence.
Never output bounding boxes. Never invent timestamps. Use the supplied absolute seconds.
confidence is an uncalibrated subjective model rating in [0,1], NOT a probability.
Return ONLY JSON, with this structure:
{{"events":[{{"observed_action":"document_handling","objects":["document"],
"evidence_timestamps":[0.0,0.5],"confidence":0.5,
"uncertainty_reason":"brief specific visibility or temporal limitation, empty if none"}}]}}
The example timestamps are placeholders: use actual supplied timestamps instead.
'''


def offline():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'


class LocalVLM:
    def __init__(self, model_path, lock, device='cuda'):
        offline()
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        path = Path(model_path).resolve()
        actual = json.loads((path / 'autolabel_model_lock.json').read_text())
        if any(actual[k] != lock[k] for k in ('model_id', 'model_revision')):
            raise ValueError('Local model provenance differs from pinned lock')
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True,
                                                       trust_remote_code=False, use_fast=False,
                                                       min_pixels=128*28*28, max_pixels=256*28*28)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
            attn_implementation='sdpa').to(device).eval()

    def predict(self, sampled):
        import torch
        content = [{'type': 'text', 'text': PROMPT.format(actions=', '.join(ACTIONS), objects=', '.join(OBJECTS))}]
        for ts, _ in sampled:
            content.extend([{'type': 'text', 'text': f'Frame at {ts:.6f} seconds:'}, {'type': 'image'}])
        messages = [{'role': 'user', 'content': content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[im for _, im in sampled], padding=True, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=650, do_sample=False)
        return self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def parse_events(text):
    text = text.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    elif text.startswith('```') and text.endswith('```'):
        text = text[3:-3].strip()
    value = json.loads(text)
    # A bare event array is a lossless wrapper variation, not a semantic repair.
    if isinstance(value, list):
        value = {'events': value}
    if not isinstance(value, dict) or set(value) != {'events'} or not isinstance(value['events'], list) or len(value['events']) > 3:
        raise ValueError('Expected JSON object with at most three events')
    expected = {'observed_action', 'objects', 'evidence_timestamps', 'confidence', 'uncertainty_reason'}
    if any(not isinstance(e, dict) or set(e) != expected for e in value['events']):
        raise ValueError('Unexpected model event fields')
    return value['events']
