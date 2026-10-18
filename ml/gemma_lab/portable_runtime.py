"""Unpack private ABI libraries; never install packages or alter the host loader."""
import argparse
import json
from pathlib import Path
import subprocess
from .bootstrap import download

PACKAGES={
 'pool/main/g/glibc/libc6_2.39-0ubuntu8.9_amd64.deb':'ff5557d99b51f761c4b7c92368b9cc45565eda17df9bf9eb4b134d09825008be',
 'pool/main/g/gcc-14/libstdc++6_14.2.0-4ubuntu2~24.04.1_amd64.deb':'a51f8de7829211db961a31f02158058ad1a95f92ac6d0a5dff6350e2821c54c0',
}


def install(root):
    assets=root/'artifacts/gemma_lab';sysroot=assets/'private-sysroot';sysroot.mkdir(parents=True,exist_ok=True)
    for remote,sha in PACKAGES.items():
        archive=assets/Path(remote).name
        download('https://archive.ubuntu.com/ubuntu/'+remote,archive,sha)
        subprocess.run(['dpkg-deb','--extract',str(archive),str(sysroot)],check=True)
    # With an explicit loader /proc/self/exe points to ld-linux. Put backend
    # discovery links beside that private loader, not anywhere in the host OS.
    library_dir=sysroot/'usr/lib/x86_64-linux-gnu'
    for library in (assets/'runtime').rglob('libggml*.so*'):
        link=library_dir/library.name
        if not link.exists():link.symlink_to(library.resolve())
    (assets/'private-runtime.json').write_text(json.dumps(dict(packages=PACKAGES,
        host_packages_changed=False,license='Ubuntu package copyright files included in private-sysroot/usr/share/doc'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);install(p.parse_args().root)
