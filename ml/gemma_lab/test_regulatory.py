import unittest
from .regulatory import source_for_assessment, regulatory_source


class RegulatoryTests(unittest.TestCase):
    def test_same_scenario_uses_object_specific_provision(self):
        rule = dict(rule_id='valuables_on_departure', evidence=[dict(observation_id='o')])
        def source(prop):
            return source_for_assessment(rule, [dict(observation_id='o', kind='state', property=prop)])
        self.assertEqual(source('key_location')['section'], 'пункт 80')
        self.assertEqual(source('stamp_location')['section'], 'пункт 80')
        self.assertEqual(source('cash_location')['section'], 'пункт 20')
        self.assertIsNone(source('document_location'))
        self.assertIsNone(source_for_assessment(rule, []))

    def test_door_and_equipment_are_context_references(self):
        for name in ('door_after_entry', 'equipment_on_departure'):
            self.assertEqual(regulatory_source(name)['qualification'], 'reference_only')
        self.assertIn('30 минут', regulatory_source('equipment_on_departure')['legal_note'])


if __name__ == '__main__':
    unittest.main()
