"""Prompt selection boundaries; does not start a model or change the server."""
from pathlib import Path
import tempfile
import unittest

from ml.gemma_lab.runner import load_instructions, MAX_INSTRUCTIONS_BYTES, prompt_guidance


class InstructionsSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root / 'ml/gemma_lab'
        self.folder.mkdir(parents=True)
        self.default = self.folder / 'instructions.txt'
        self.default.write_text('Existing production instructions\n')

    def test_default_is_unchanged_and_override_is_explicit(self):
        text, path = load_instructions(self.root)
        self.assertEqual(text, self.default.read_text())
        self.assertEqual(path, self.default)
        candidate = self.folder / 'adaptation/instructions_v3.txt'
        candidate.parent.mkdir()
        candidate.write_text('Candidate, not activated\n')
        selected, source = load_instructions(self.root, 'ml/gemma_lab/adaptation/instructions_v3.txt')
        self.assertEqual(selected, candidate.read_text())
        self.assertEqual(source, candidate)
        self.assertEqual(load_instructions(self.root)[0], text)

    def test_foreign_absolute_and_traversal_paths_are_rejected(self):
        outside = self.root / 'unrelated.txt'
        outside.write_text('Do not load')
        for path in [outside, Path('ml/gemma_lab/../../unrelated.txt')]:
            with self.assertRaises(ValueError):
                load_instructions(self.root, path)

    def test_symlink_cannot_escape_component(self):
        outside = self.root / 'unrelated.txt'
        outside.write_text('Do not load')
        link = self.folder / 'escape.txt'
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            load_instructions(self.root, link)

    def test_wrong_type_blank_and_oversize_are_rejected(self):
        for name, text in [('instructions.py', 'print(1)'), ('empty.txt', ' \n'),
                           ('large.txt', 'x' * (MAX_INSTRUCTIONS_BYTES + 1))]:
            path = self.folder / name
            path.write_text(text)
            with self.assertRaises(ValueError):
                load_instructions(self.root, path)

    def test_default_guidance_keeps_preexperiment_wording(self):
        selected = prompt_guidance()
        self.assertEqual(selected['version'], 'legacy_default_v1')
        self.assertEqual(selected['current_pass_suffix'],
                         '. Return salient events, states, relevant rule_checks and unknowns. Keep JSON compact.')
        self.assertEqual(selected['final'],
                         'Return ONLY {"events":[],"states":[],"rule_checks":[]} with current visual evidence. '
                         'Never output or copy the input memory schema. Use short Russian descriptions and at most 3 evidence IDs per observation. '
                         'At most 3 events, 3 states and 2 rule checks. Empty arrays are valid when nothing can be established.')

    def test_only_explicit_override_uses_observation_first_guidance(self):
        selected = prompt_guidance(Path('ml/gemma_lab/adaptation/instructions_v4.txt'))
        self.assertEqual(selected['version'], 'observation_first_explicit_v4')
        self.assertIn('First record supported visible actions and states', selected['current_pass_suffix'])
        self.assertNotIn('[]', selected['final'])
        self.assertIn('no minimum quota', selected['final'])
        # Passing an explicit path is the opt-in, not mutating the default file.
        self.assertEqual(prompt_guidance(Path('ml/gemma_lab/instructions.txt'))['version'], selected['version'])
        self.assertEqual(prompt_guidance()['version'], 'legacy_default_v1')


if __name__ == '__main__':
    unittest.main()
