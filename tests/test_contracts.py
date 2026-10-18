"""Portable checks for the published contract, no model or banking data."""
import json
from pathlib import Path
import unittest


class JuryContractTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads((Path(__file__).resolve().parents[1] /
                                    'contracts/jury.openapi.json').read_text())

    def test_routes_and_key_keep_the_existing_public_contract(self):
        self.assertEqual(set(self.document['paths']), {
            '/api/v1/analyze', '/api/v1/jobs/{job_id}', '/api/v1/jobs/{job_id}/preview'})
        self.assertEqual(self.document['components']['securitySchemes']['JuryKey'],
                         dict(type='apiKey', **{'in':'header'}, name='X-API-Key'))
        self.assertEqual(self.document['security'], [{'JuryKey': []}])

    def test_terminal_states_and_evidence_constraints_are_explicit(self):
        schemas = self.document['components']['schemas']
        self.assertEqual(schemas['Completed']['properties']['status'], {'const':'done'})
        self.assertEqual(schemas['Failed']['properties']['status'], {'const':'failed'})
        self.assertEqual(schemas['Completed']['properties']['verdict']['enum'],
                         ['no_violation', 'violation'])
        frames = schemas['Violation']['properties']['evidence_frames']
        self.assertEqual((frames['minItems'],frames['maxItems']), (1,3))
        self.assertEqual(schemas['EvidenceFrame']['properties']['image_base64']['contentMediaType'], 'image/jpeg')

    def test_all_local_schema_references_resolve(self):
        def walk(value):
            if isinstance(value, dict):
                if '$ref' in value:
                    target = self.document
                    self.assertTrue(value['$ref'].startswith('#/'))
                    for part in value['$ref'][2:].split('/'):
                        target = target[part]
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(self.document)
