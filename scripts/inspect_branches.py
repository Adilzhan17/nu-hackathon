"""Read-only branch inventory; never fetch, rename, merge, delete or checkout."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def inventory(base='origin/main'):
    base_sha = git('rev-parse', '--verify', '--end-of-options', base + '^{commit}')
    records = []
    for line in git('for-each-ref', '--format=%(refname:short)\t%(objectname)\t%(upstream:short)\t%(symref)',
                    'refs/heads', 'refs/remotes/origin').splitlines():
        parts = line.split('\t')
        if len(parts) > 3 and parts[3]:
            continue  # origin/HEAD is an alias, not another development branch.
        name, revision = parts[:2]
        behind, ahead = map(int, git('rev-list', '--left-right', '--count', base_sha + '...' + revision).split())
        records.append(dict(name=name, revision=revision, upstream=parts[2] if len(parts)>2 else None,
                            ahead_of_base=ahead, behind_base=behind))
    return dict(base=base, base_revision=base_sha, current=git('branch', '--show-current'), branches=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='origin/main')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    report = inventory(args.base)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print('Current:', report['current'], '| Comparison base:', report['base'])
        for row in report['branches']:
            print(f"{row['name']:42} {row['revision'][:8]} +{row['ahead_of_base']} -{row['behind_base']}")


if __name__ == '__main__':
    main()
