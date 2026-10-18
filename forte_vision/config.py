"""Infrastructure settings, independent of model, data and HTTP handlers.

Existing deployment defaults are retained during the compatibility migration.
New installations override these values through the environment; no keys are
read here and no network operation occurs while importing configuration.
"""
from dataclasses import dataclass
import os
from pathlib import PurePosixPath
import re


@dataclass(frozen=True)
class GPUTransport:
    host: str
    port: int
    key_file: str
    root: str

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        host = env.get('FORTE_GPU_HOST', 'ubuntu@global.prd.ga.run.brev.nvidia.com')
        port = int(env.get('FORTE_GPU_PORT', '15171'))
        key = env.get('FORTE_GPU_KEY_FILE', '/root/.ssh/nuhack_brev')
        root = env.get('FORTE_GPU_ROOT', '/home/ubuntu/nuhack-training')
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', host):
            raise ValueError('FORTE_GPU_HOST must be an SSH host, not a command or URL')
        if not 1 <= port <= 65535:
            raise ValueError('FORTE_GPU_PORT must be between 1 and 65535')
        for name, value in [('FORTE_GPU_KEY_FILE', key), ('FORTE_GPU_ROOT', root)]:
            path = PurePosixPath(value)
            if not path.is_absolute() or '..' in path.parts or str(path) == '/' or any(c in value for c in '\n\r\x00'):
                raise ValueError(name + ' must be a non-root absolute path without traversal')
        return cls(host=host, port=port, key_file=key, root=root.rstrip('/'))

    def ssh_command(self):
        return ['ssh', '-i', self.key_file, '-p', str(self.port), '-o',
                'BatchMode=yes', '-o', 'ConnectTimeout=5', self.host]
