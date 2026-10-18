"""Blue/green local frontend switch. Annotation listener and old frontend stay up."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path('/root/nuhack')
FLAG=ROOT/'runs/forte_x/active'
RULE=['-p','tcp','-d','127.0.0.1','--dport','3000','-m','comment','--comment','forte-x-frontend-v1','-j','REDIRECT','--to-ports','3001']
def installed():
    return subprocess.run(['iptables','-t','nat','-C','OUTPUT',*RULE],capture_output=True).returncode==0
def health():
    for _ in range(30):
        try:
            with urllib.request.urlopen('http://127.0.0.1:3001/analysis/new',timeout=2) as response:
                if response.status==200 and b'Forte X' in response.read():return
        except OSError:pass
        time.sleep(.5)
    raise RuntimeError('New frontend failed readiness; old frontend retained')
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['ready','activate','deactivate','rollback']);a=p.parse_args()
    if a.action in ['ready','activate'] and (ROOT/'ml/forte_x/UI_HANDOFF.txt').exists():
        raise RuntimeError('Retired parallel frontend. The current single UI is /analyze; see UI_HANDOFF.txt. Do not reactivate port3001 routing.')
    if a.action in ['ready','activate']:
        health()
        if a.action=='activate':FLAG.write_text('Forte X v1\n')
        if FLAG.exists() and not installed():subprocess.run(['iptables','-t','nat','-I','OUTPUT','1',*RULE],check=True)
    else:
        if installed():subprocess.run(['iptables','-t','nat','-D','OUTPUT',*RULE],check=True)
        if a.action=='rollback':FLAG.unlink(missing_ok=True)
    print(json.dumps(dict(action=a.action,active=installed(),old_frontend_preserved=True,annotation_restarted=False)))
if __name__=='__main__':main()
