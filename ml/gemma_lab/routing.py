"""Reversible loopback handover; leave the original web and annotator processes alive."""
import argparse
import subprocess
import time
import urllib.request

RULE=['OUTPUT','-p','tcp','-d','127.0.0.1','--dport','3000','-m','comment','--comment','gemma-lab-web-v1','-j','REDIRECT','--to-ports','3002']
def change(enable):
    if enable:
        for attempt in range(40):
            try:
                with urllib.request.urlopen('http://127.0.0.1:3002/demo/gemma',timeout=2) as r:
                    if r.status==200 and 'Gemma 4' in r.read().decode():break
            except OSError:pass
            time.sleep(1)
        else:raise RuntimeError('Gemma web is not healthy; original routing preserved')
    exists=subprocess.run(['/usr/sbin/iptables','-w','5','-t','nat','-C',*RULE],capture_output=True).returncode==0
    if exists!=enable:
        subprocess.run(['/usr/sbin/iptables','-w','5','-t','nat','-A' if enable else '-D',*RULE],check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['enable','disable']);change(p.parse_args().action=='enable')
