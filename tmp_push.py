import os
import subprocess
import sys

repo = 'c:/Users/Alim1/OneDrive/Desktop/project walmart'
os.chdir(repo)

files = [
    'Dockerfile',
    '.github/workflows/ci.yml',
    'readme.md',
    'api/main.py',
    'tests/test_api.py',
    'src/train.py',
    'tests/test_train.py',
]

for cmd in [
    ['git', 'status', '--short'],
    ['git', 'add', *files],
    ['git', 'commit', '-m', 'Add Phase 5 and Phase 6 deployment setup'],
    ['git', 'push', 'origin', 'main'],
]:
    print('$', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f'Command failed with exit code {result.returncode}')
        sys.exit(result.returncode)
