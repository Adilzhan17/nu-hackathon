"""Prepare a human-only action pilot from the existing frozen fit partition.

No external team's code, weights, annotations or clip-specific hints are used.
The generic train prompt matches the existing paired action evaluation prompt.
This changes action supervision, not video labels or the production model.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .eval_prompts import prompt
from .taxonomy import ACTION_CODES
from .train import validate_rows


def select(rows, protocol):
    validate_rows(rows, protocol)
    selected, excluded = [], []
    for row in rows:
        target = json.loads(row['target'])
        action = target.get('observed_action')
        if row['kind'] != 'human_action':
            reason = 'unconfirmed_model_caption'
        elif action in {'other_observed_action', 'cash_visible', 'safe_open_visible'}:
            reason = 'nonspecific_action_or_state'
        elif action not in ACTION_CODES:
            reason = 'unknown_action'
        else:
            instruction = prompt('base_v2prompt', 'action', row['frames'], ACTION_CODES)
            selected.append(dict(row, prompt=instruction,
                                 prompt_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                                 experiment='human_only_aligned_v1'))
            continue
        excluded.append(dict(sample_id=row['sample_id'], reason=reason))
    validate_rows(selected, protocol)
    return selected, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--learning-rate', type=float, default=0.00005)
    args = parser.parse_args()
    source = args.run_root / 'prepared/training_samples_v2.jsonl'
    protocol_path = args.run_root / 'frozen_group_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    rows, excluded = select([json.loads(line) for line in source.read_text().splitlines() if line.strip()], protocol)
    unique_count = len(rows)
    if args.repeats > 1:
        rows = [dict(row, sample_id=f"{row['sample_id']}:exposure{exposure}",
                     source_sample_id=row['sample_id'], exposure_index=exposure)
                for exposure in range(args.repeats) for row in rows]
        validate_rows(rows, protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / 'training_samples.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    report = dict(schema_version=1, experiment='human_only_aligned_v1', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  protocol_sha256=protocol['protocol_sha256'], samples=len(rows), unique_human_examples=unique_count,
                  repetitions_per_example=args.repeats, groups=len({r['group_id'] for r in rows}),
                  actions=dict(Counter(json.loads(r['target'])['observed_action'] for r in rows)), excluded=excluded,
                  dataset_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  evaluation='Existing 16 eligible inner-holdout action intervals, paired base_v2prompt/lora_v2prompt; then the same 8 inner-holdout videos.',
                  limitation='Repeated development holdout, not independent test; action top-1 is not video violation accuracy.',
                  recipe=dict(seed=7301, learning_rate=args.learning_rate, max_updates=len(rows),
                              shuffled_exposures_per_unique_example=args.repeats),
                  production_activation=False)
    (args.output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('samples', 'groups', 'actions', 'dataset_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
