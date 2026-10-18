import unittest

from forte_vision.config import GPUTransport


class GPUTransportTests(unittest.TestCase):
    def test_defaults_preserve_existing_deployment(self):
        config = GPUTransport.from_env({})
        self.assertEqual(config.port, 15171)
        self.assertEqual(config.root, '/home/ubuntu/nuhack-training')
        self.assertEqual(config.ssh_command()[-1], config.host)

    def test_new_deployment_can_override_every_host_specific_value(self):
        config = GPUTransport.from_env(dict(FORTE_GPU_HOST='worker@gpu.internal',
            FORTE_GPU_PORT='22', FORTE_GPU_KEY_FILE='/run/secrets/gpu_key',
            FORTE_GPU_ROOT='/opt/forte vision/'))
        self.assertEqual(config.root, '/opt/forte vision')
        self.assertEqual(config.ssh_command(), ['ssh', '-i', '/run/secrets/gpu_key',
            '-p', '22', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'worker@gpu.internal'])

    def test_invalid_config_fails_before_starting_any_process(self):
        for key, values in {
            'FORTE_GPU_HOST': ['-oProxyCommand=evil', 'worker@host; touch file', 'https://host'],
            'FORTE_GPU_PORT': ['0', '65536', 'port'],
            'FORTE_GPU_KEY_FILE': ['relative/key', '/', '/a/../secret', '/key\nfile'],
            'FORTE_GPU_ROOT': ['.', '/', '/opt/../', '/opt/run\x00'],
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    GPUTransport.from_env({key: value})
