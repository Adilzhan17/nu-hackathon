"""Dependency and migration contracts: pure layers cannot import experiments."""
import ast
import hashlib
import importlib
from importlib.util import resolve_name
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_analysis_screen_has_one_frontend_owner(self):
        self.assertTrue((ROOT / 'frontend/analysis/index.html').is_file())
        self.assertFalse((ROOT / 'ml/gemma_lab/index.html').exists())

    def test_domain_and_application_have_only_inward_dependencies(self):
        allowed = {
            'domain': {'domain'},
            'application': {'domain', 'application'},
            'inference': {'domain', 'application', 'config'},
            'infrastructure': {'domain', 'application', 'config', 'infrastructure'},
        }
        for layer, permitted in allowed.items():
            for path in (ROOT / 'forte_vision' / layer).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    imports = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                    if isinstance(node, ast.ImportFrom):
                        package = '.'.join(path.relative_to(ROOT).parts[:-1])
                        base = resolve_name('.' * node.level + (node.module or ''), package)
                        imports = [base, *(base + '.' + alias.name for alias in node.names)]
                    for name in imports:
                        with self.subTest(file=str(path.relative_to(ROOT)), dependency=name):
                            self.assertFalse(name == 'ml' or name.startswith('ml.'))
                            if name.startswith('forte_vision.'):
                                self.assertIn(name.split('.')[1], permitted)
                            if layer in {'domain', 'application'}:
                                self.assertNotIn(name.split('.')[0],
                                    {'torch', 'transformers', 'av', 'subprocess', 'socket', 'urllib', 'requests'})

    def test_production_import_does_not_load_research_or_training_modules(self):
        code = '''
import sys
import forte_vision.interfaces.analysis_http
import forte_vision.inference.gemma
assert not any(n == "ml" or n.startswith("ml.") for n in sys.modules)
assert not any(n in sys.modules for n in ("torch", "transformers", "peft"))
'''
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True,
                       capture_output=True, timeout=15)

    def test_old_imports_resolve_to_the_same_implementation(self):
        modules = {
            'ml.gemma_lab.protocol': 'forte_vision.application.analysis',
            'ml.gemma_lab.regulatory': 'forte_vision.domain.regulations',
            'ml.gemma_lab.schema': 'forte_vision.application.schema',
            'ml.gemma_lab.runner': 'forte_vision.inference.gemma',
            'ml.gemma_lab.bridge': 'forte_vision.infrastructure.gpu_transport',
            'ml.gemma_lab.server': 'forte_vision.interfaces.analysis_http',
            'ml.context_pipeline.rules': 'forte_vision.domain.scenarios',
            'ml.narration_training.organizer.context': 'forte_vision.domain.observations',
        }
        for legacy, canonical in modules.items():
            with self.subTest(legacy=legacy):
                self.assertIs(importlib.import_module(legacy), importlib.import_module(canonical))

    def test_legacy_script_import_with_only_ml_on_path_still_works(self):
        code = f'''import sys
sys.path.insert(0, {str(ROOT / 'ml')!r})
from gemma_lab.regulatory import regulatory_source
assert regulatory_source('valuables_on_departure')['section'] == 'пункт 80'
'''
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, '-c', code], cwd=directory, check=True,
                           capture_output=True, timeout=15)

    def test_production_prompt_preserves_frozen_experiment_bytes(self):
        production = ROOT / 'forte_vision/inference/prompts/scenario_v4.txt'
        historical = ROOT / 'ml/gemma_lab/adaptation/instructions_v4.txt'
        self.assertEqual(hashlib.sha256(production.read_bytes()).digest(),
                         hashlib.sha256(historical.read_bytes()).digest())
        from forte_vision.inference.gemma import load_instructions
        self.assertEqual(load_instructions(ROOT, production)[0], production.read_text())

    def test_prompt_symlinks_cannot_escape_either_registered_directory(self):
        from forte_vision.inference.gemma import load_instructions
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / 'forte_vision/inference/prompts'
            prompts.mkdir(parents=True)
            outside = root / 'private.txt'
            outside.write_text('not a prompt')
            (prompts / 'escape.txt').symlink_to(outside)
            with self.assertRaises(ValueError):
                load_instructions(root, prompts / 'escape.txt')

    def test_stable_service_entrypoints_offer_help_without_loading_models(self):
        for name in ('forte_vision.interfaces.analysis_http',
                     'forte_vision.interfaces.jury_http',
                     'forte_vision.interfaces.review_http',
                     'forte_vision.inference.gemma',
                     'ml.gemma_lab.server', 'ml.gemma_lab.runner'):
            with self.subTest(module=name):
                result = subprocess.run([sys.executable, '-m', name, '--help'], cwd=ROOT,
                    check=True, capture_output=True, text=True, timeout=15)
                self.assertIn('usage:', result.stdout)
