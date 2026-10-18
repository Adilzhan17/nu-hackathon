import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .report import checkpoint_details, effects, evaluation_summary, local_path, training_summary, forced_choice_summary, format_outputs, sha
from .publish_status import render


def dump(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in value) if jsonl else json.dumps(value))


class ReportTests(unittest.TestCase):
    def _forced_fixture(self, root):
        samples = [dict(sample_id=f's{i}', group_id=f'g{i}', video=f'v{i}', video_sha256=f'sha{i}', split='train',
                        task='video', partition='inner_holdout', frames=[dict(sha256=f'f{i}_{j}') for j in range(16)],
                        reference=dict(label=i % 2)) for i in range(8)]
        protocol = dict(inner_holdout_groups=[f'g{i}' for i in range(8)], diagnostic_val_groups=[], original_manifest_sha256='original',
                        source_identity={f'g{i}': dict(video_sha256=f'sha{i}', split='train') for i in range(8)})
        dump(root / 'frozen_group_protocol.json', protocol)
        dump(root / 'prepared/eval_video_samples.jsonl', samples, True)
        folder = root / 'forced'
        config = dict(protocol='gemma_all16_digit_logits_diagnostic_v1', additional_protocol_not_replacement=True,
                      frozen_protocol_file_sha256=sha(root / 'frozen_group_protocol.json'), original_manifest_sha256='original',
                      samples_sha256=sha(root / 'prepared/eval_video_samples.jsonl'), partition='inner_holdout',
                      sample_ids=[sample['sample_id'] for sample in samples],
                      frame_sha256={sample['sample_id']: [frame['sha256'] for frame in sample['frames']] for sample in samples},
                      organizer_answers_in_prompt=False, threshold_tuned=False, scores_are_calibrated_probabilities=False,
                      adapter_checksums={'adapter_model.safetensors': 'weights'})
        records = [dict(sample_id=sample['sample_id'], group_id=sample['group_id'], condition=condition,
                        predicted_label=sample['reference']['label'] if condition == 'base_same_prompt' else 0,
                        candidate_logits={'0': 0., '1': 1.} if condition == 'base_same_prompt' and sample['reference']['label'] == 1 else {'0': 1., '1': 0.},
                        decision_scope='whole_video', organizer_answers_in_prompt=False,
                        frame_sha256=config['frame_sha256'][sample['sample_id']], prompt_sha256='same',
                        scores_are_calibrated_probabilities=False, status='completed')
                   for sample in samples for condition in ('base_same_prompt', 'lora_same_prompt')]
        dump(folder / 'config.json', config)
        dump(folder / 'predictions.jsonl', records, True)
        dump(folder / 'status.json', dict(status='completed'))
        return folder, config, records

    def test_additional_forced_protocol_checks_all_eight_and_recomputes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, config, rows = self._forced_fixture(root)
            result = forced_choice_summary(root, folder, dict(available=True, adapter_sha256='weights'))
            self.assertEqual(result['fully_paired_groups'], 8)
            self.assertTrue(result['comparison_complete'])
            self.assertEqual(result['paired_only_conditions']['base_same_prompt']['correct_fraction_all_groups'], 1.)
            self.assertEqual(result['paired_only_conditions']['lora_same_prompt']['correct_fraction_all_groups'], .5)
            self.assertFalse(result['event_type_accuracy_measured'])
            self.assertEqual(result['effect']['conclusion'], 'regression_in_this_additional_diagnostic')
            rows[1]['prompt_sha256'] = 'different'
            dump(folder / 'predictions.jsonl', rows, True)
            with self.assertRaisesRegex(ValueError, 'prompt'):
                forced_choice_summary(root, folder, dict(available=True, adapter_sha256='weights'))
            dump(folder / 'predictions.jsonl', rows[2:], True)
            config['sample_ids'] = config['sample_ids'][1:]
            dump(folder / 'config.json', config)
            with self.assertRaisesRegex(ValueError, 'schedule'):
                forced_choice_summary(root, folder, dict(available=True, adapter_sha256='weights'))

    def test_additional_protocol_does_not_override_action_regression_or_render_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, _, _ = self._forced_fixture(root)
            forced = forced_choice_summary(root, folder, dict(available=True, adapter_sha256='weights'))
            action = dict(paired_only_conditions={condition: dict(inner_holdout=dict(supplied_action_intervals=dict(correct=correct, scorable=16,
                                         top1_correct_fraction_all=correct / 16))) for condition, correct in [('base_v2prompt', 2), ('lora_v2prompt', 1)]})
            summary = dict(training=dict(used_by_kind={}, status='completed', actual_unique_samples=86, checkpoint=dict(available=False)),
                           data=dict(eligible_human=41, eligible_weak=45, eligible_total=86, fit_groups=24, fit_groups_with_samples=10,
                                     inner_holdout_groups=8, diagnostic_val_groups=8), training_history=[], video_evaluation={},
                           action_evaluation=action, effects=effects({}, action), forced_choice_evaluation=forced)
            report, public = format_outputs(summary)
            self.assertIn('2/16', report)
            self.assertIn('1/16', report)
            self.assertIn('Дополнительный протокол', report)
            self.assertEqual(summary['effects']['conditional_action_top1']['absolute_change'], -1 / 16)
            public['additional_protocol']['message'] = '<script>bad</script>'
            html = render(public)
            self.assertIn('2/16', html)
            self.assertIn('1/16', html)
            self.assertIn('&lt;script&gt;', html)
            self.assertNotIn('<script>', html)
            self.assertIn('не заменяет основные результаты', html)

    def test_remote_checkpoint_resolution_and_escape_guard(self):
        root = Path('/tmp/report_test/adaptation_v1')
        actual = local_path('/home/ubuntu/nuhack-training/runs/gemma_lab/adaptation_v1/train/checkpoints/update_000002', root)
        self.assertEqual(actual, root / 'train/checkpoints/update_000002')
        with self.assertRaises(ValueError):
            local_path('/etc/passwd', root)

    def test_training_counts_only_successful_actual_ledger_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'train'
            eligible = [dict(sample_id='a', group_id='g', video_sha256='s', partition='fit', kind='human_action'),
                        dict(sample_id='b', group_id='g', video_sha256='s', partition='fit', kind='weak_caption')]
            good = dict(sample_id='a', group_id='g', video_sha256='s', kind='human_action', result=dict(updated=True))
            bad = dict(sample_id='b', group_id='g', video_sha256='s', kind='weak_caption', result=dict(updated=False, reason='encoding_rejected'))
            dump(folder / 'consumption.jsonl', [good, bad], True)
            dump(folder / 'status.json', dict(status='partial', optimizer_updates=1, consumed_records=[good],
                                             consumed_sample_ids=['a'], checkpoint=None))
            report, history = training_summary(root, folder, eligible)
            self.assertEqual(report['actual_unique_samples'], 1)
            self.assertEqual(report['used_by_kind'], {'human_action': 1})
            self.assertEqual(report['skipped_reasons'], {'encoding_rejected': 1})
            self.assertEqual(report['samples_not_used'], 1)
            self.assertFalse(report['checkpoint']['available'])
            self.assertEqual(len(history), 1)

    def test_unpaired_video_not_credited_and_prompt_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'eval'
            samples = [dict(sample_id='a', group_id='g1', video='a', task='video', partition='inner_holdout',
                            frames=[dict(sha256='f')], reference=dict(label=1)),
                       dict(sample_id='b', group_id='g2', video='b', task='video', partition='inner_holdout',
                            frames=[dict(sha256='h')], reference=dict(label=0))]
            manifest = root / 'prepared/samples.jsonl'
            dump(manifest, samples, True)
            config = dict(manifests={'prepared/samples.jsonl': hashlib.sha256(manifest.read_bytes()).hexdigest()},
                          sample_ids=['a', 'b'], adapter_checksums={'adapter_model.safetensors': 'weights'})
            dump(folder / 'config.json', config)
            records = [dict(sample_id=sid, group_id=gid, task='video', partition='inner_holdout', condition=condition,
                            decision_scope='whole_video', organizer_answers_in_prompt=False, frame_sha256=[frame],
                            prompt_sha256=['same'], predicted_label=label, rule_verdict='insufficient_data',
                            status='completed', parse_status='accepted') for sid, gid, frame, label, condition in
                       [('a', 'g1', 'f', 1, 'base_v2prompt'), ('a', 'g1', 'f', 0, 'lora_v2prompt'),
                        ('b', 'g2', 'h', 0, 'base_v2prompt')]]
            dump(folder / 'predictions.jsonl', records, True)
            result = evaluation_summary(root, folder, {'adapter_sha256': 'weights'})
            self.assertEqual(result['fully_paired_samples'], 1)
            self.assertEqual(result['scheduled_samples'], 2)
            report = effects(result, {})
            self.assertEqual(report['whole_video_tentative']['conclusion'], 'regression_in_this_small_paired_diagnostic')
            records[1]['prompt_sha256'] = ['changed']
            dump(folder / 'predictions.jsonl', records, True)
            with self.assertRaisesRegex(ValueError, 'prompts'):
                evaluation_summary(root, folder, {'adapter_sha256': 'weights'})

    def test_changed_B_parameters_do_not_imply_all_A_parameters_changed(self):
        try:
            import numpy as np
            from safetensors.numpy import save_file
        except ImportError:
            self.skipTest('Optional CPU safetensors/numpy unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file({'module.lora_A.weight': np.ones((2, 3), dtype=np.float32),
                       'module.lora_B.weight': np.array([[0., 1.], [2., 0.]], dtype=np.float32)}, str(root / 'adapter_model.safetensors'))
            dump(root / 'adapter_config.json', dict(init_lora_weights=True, use_dora=False))
            result = checkpoint_details(root)
            self.assertEqual(result['changed_parameter_elements'], 2)
            self.assertFalse(result['exact_all_parameter_diff_available'])
            self.assertEqual(result['adapter_parameter_elements'], 10)


if __name__ == '__main__':
    unittest.main()
