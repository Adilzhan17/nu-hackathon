"""Freeze seeded, label/annotation-stratified inner holdout; never use model scores."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .audit_data import canonical_sha, write_new


def build_protocol(targets, human, seed=7301):
    by_group = {row['group_id']: row for row in targets}
    if len(by_group) != len(targets):
        raise ValueError('Duplicate whole-video groups')
    annotated = {row['group_id'] for row in human}
    if any(by_group[g]['split'] != 'train' for g in annotated):
        raise ValueError('Human action outside frozen train')
    train = {g for g, row in by_group.items() if row['split'] == 'train'}
    diagnostic = set(by_group) - train
    if len(train) != 32 or len(diagnostic) != 8:
        raise ValueError('This version requires original 32 train / 8 diagnostic groups')
    rng = random.Random(seed)
    heldout, stratum_counts = [], {}
    for label in (0, 1):
        strata = {has_human: sorted(g for g in train if by_group[g]['label'] == label and (g in annotated) == has_human)
                  for has_human in (False, True)}
        total = sum(len(items) for items in strata.values())
        if total < 4:
            raise ValueError('Cannot allocate four groups of each whole-video label')
        for items in strata.values():
            rng.shuffle(items)
        # Proportional annotation-status allocation, keeping an annotated group per
        # label where available. This choice is fixed before looking at predictions.
        human_count = round(4 * len(strata[True]) / total)
        human_count = max(1 if strata[True] else 0, human_count, 4 - len(strata[False]))
        human_count = min(len(strata[True]), 4, human_count)
        selected = strata[True][:human_count] + strata[False][:4 - human_count]
        if len(selected) != 4:
            raise ValueError('Invalid stratum allocation')
        heldout.extend(selected)
        stratum_counts[str(label)] = dict(total=total, annotated=len(strata[True]),
                                         heldout=4, heldout_annotated=human_count)
    heldout = set(heldout)
    if len(heldout & annotated) < 2:
        raise ValueError('Need at least two human-annotated inner-holdout groups')
    fit = train - heldout
    partitions = {g: 'fit' if g in fit else 'inner_holdout' if g in heldout else 'diagnostic_val' for g in sorted(by_group)}
    result = dict(schema_version=1, seed=seed, algorithm='sorted_groups_random.Random_seed_shuffle_per_label_and_has_human_actions',
                  original_split_only=True, original_manifest='data/training/manifest.csv',
                  fit_groups=sorted(fit), inner_holdout_groups=sorted(heldout), diagnostic_val_groups=sorted(diagnostic),
                  group_membership=partitions, stratum_counts=stratum_counts,
                  fit_groups_count=len(fit), inner_holdout_groups_count=len(heldout), diagnostic_val_groups_count=len(diagnostic),
                  inner_holdout_human_groups=sorted(heldout & annotated),
                  fit_label_counts=dict(Counter(str(by_group[g]['label']) for g in fit)),
                  inner_holdout_label_counts=dict(Counter(str(by_group[g]['label']) for g in heldout)),
                  source_identity={g: {k: by_group[g][k] for k in ('video', 'video_sha256', 'split', 'label')} for g in sorted(by_group)},
                  input_targets_sha256=canonical_sha(targets), input_human_rows_sha256=canonical_sha(human),
                  policy=dict(model_results_used_for_selection=False, diagnostic_val_used_for_training=False,
                              diagnostic_val_used_for_threshold_selection=False, diagnostic_val_is_independent_test=False,
                              exclude_inner_holdout_from_all_training_sources=True,
                              inner_holdout_captions_must_not_appear_in_model_prompts=True,
                              source_groups_apply_to_aliases_crops_synthetic_frames_and_cached_features=True,
                              human_annotations_may_be_sparse=True))
    result['protocol_sha256'] = canonical_sha(result)
    return result


def run(dataset):
    dataset = dataset.resolve()
    def rows(name):
        return [json.loads(line) for line in (dataset / name).read_text().splitlines() if line.strip()]
    targets = rows('video_level_targets.jsonl')
    human = rows('human_actions_train.jsonl')
    protocol = build_protocol(targets, human)
    audit = json.loads((dataset / 'audit.json').read_text())
    protocol['original_manifest_sha256'] = audit['protected_input_sha256']['data/training/manifest.csv']
    # Recompute after adding immutable source digest.
    protocol.pop('protocol_sha256')
    protocol['protocol_sha256'] = canonical_sha(protocol)
    write_new(dataset / 'frozen_group_protocol.json', protocol)
    counts, exports = {}, {}
    for source in ('human_actions', 'weak_pairs', 'human_objects', 'organizer_claims'):
        incoming = rows(source + '_train.jsonl')
        for partition in ('fit', 'inner_holdout'):
            selected = [dict(row, adaptation_partition=partition) for row in incoming
                        if protocol['group_membership'][row['group_id']] == partition]
            name = source + '_' + partition + '.jsonl'
            write_new(dataset / name, selected, jsonl=True)
            counts[name] = dict(rows=len(selected), groups=len({r['group_id'] for r in selected}))
            exports[name] = hashlib.sha256((dataset / name).read_bytes()).hexdigest()
    result = dict(counts=counts, output_sha256=exports,
                  protocol_sha256=protocol['protocol_sha256'],
                  holdout_inputs_are_for_scoring_only_not_training_or_prompts=True)
    write_new(dataset / 'partition_export_audit.json', result)
    print(json.dumps(dict(partitions=counts, strata=protocol['stratum_counts'],
                          inner_holdout_groups=[protocol['source_identity'][g]['video'] for g in protocol['inner_holdout_groups']]),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset)
