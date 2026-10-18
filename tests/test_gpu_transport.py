"""Dry-run the coordinator transport without SSH, decoding or GPU inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from forte_vision.infrastructure import gpu_transport


class TransportContractTests(unittest.TestCase):
    def test_remote_command_uses_canonical_worker_and_quoted_configured_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / ('a' * 32)
            (folder / 'remote').mkdir(parents=True)
            (folder / 'remote/result.json').write_text(json.dumps(dict(status='completed', stage='done')))
            plan = dict(duration_sec=4, frames=[dict(timestamp=0), dict(timestamp=.5)])
            process = Mock()
            process.poll.return_value = 0
            emit = Mock()
            ssh = ['ssh', '-i', '/private/key with spaces', 'worker@gpu.internal']
            with patch.object(gpu_transport, 'prepare', return_value=plan), \
                 patch.object(gpu_transport, 'SSH', ssh), \
                 patch.object(gpu_transport, 'REMOTE', '/opt/forte vision'), \
                 patch.object(gpu_transport.subprocess, 'run') as execute, \
                 patch.object(gpu_transport.subprocess, 'Popen', return_value=process) as launch:
                gpu_transport.process(folder, emit, seconds=270)
            command = launch.call_args.args[0]
            self.assertEqual(command[:-1], ssh)
            self.assertIn("cd '/opt/forte vision'", command[-1])
            self.assertIn('-m forte_vision.inference.gemma', command[-1])
            self.assertIn('--instructions forte_vision/inference/prompts/scenario_v4.txt', command[-1])
            self.assertIn('timeout -k 10 295', command[-1])
            transfers = [call.args[0] for call in execute.call_args_list if call.args[0][0] == 'rsync']
            self.assertEqual(len(transfers), 2)
            self.assertTrue(all('--protect-args' in call for call in transfers))
            self.assertEqual(emit.call_args.kwargs['status'], 'completed')
            process.terminate.assert_not_called()
