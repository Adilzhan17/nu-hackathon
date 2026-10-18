from pathlib import Path
import unittest

from ml.gemma_lab.adaptation.check_instruction_compatibility import contract_check
from ml.gemma_lab.runner import OUTPUT_GUIDANCE, load_instructions
from ml.gemma_lab.schema import output_schema


class InstructionV4Tests(unittest.TestCase):
    def test_new_instruction_contract_and_size(self):
        root = Path(__file__).resolve().parents[3]
        path = root / 'ml/gemma_lab/adaptation/instructions_v4.txt'
        text, selected = load_instructions(root, path)
        self.assertEqual(selected, path)
        self.assertGreaterEqual(len(text.encode()), 4096)
        self.assertLessEqual(len(text.encode()), 6144)
        schema = output_schema([dict(id='f0', timestamp=0), dict(id='f1', timestamp=.5)])
        self.assertTrue(contract_check(path, schema)['structural_contract_compatible'])
        self.assertNotIn('{"events":[]', text)
        self.assertNotIn('{"events":[]', OUTPUT_GUIDANCE)
        self.assertNotIn('[]', OUTPUT_GUIDANCE)

    def test_schema_still_permits_honest_empty_output(self):
        schema = output_schema([dict(id='f0', timestamp=0), dict(id='f1', timestamp=.5)])
        for name in ('events', 'states', 'rule_checks'):
            self.assertEqual(schema['properties'][name]['minItems'], 0)
        self.assertIn('no minimum quota', OUTPUT_GUIDANCE)
        self.assertIn('omit unsupported', OUTPUT_GUIDANCE)


if __name__ == '__main__':
    unittest.main()
