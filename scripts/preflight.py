#!/usr/bin/env python
"""Check the repository is submission-ready before an entry is spent.

A failed entry does not count against the ten official-phase submissions, but a
round trip still costs up to 72 hours of feedback latency, and there are four
weeks left. Everything here is something that has actually gone wrong or was one
step away from going wrong in this repo.

Usage:
    python scripts/preflight.py
"""

import ast
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The organizers substitute their own copies of these, so any local edit is
# invisible in testing and active in scoring.
ORGANIZER_OWNED = ['helper_code.py', 'train_model.py', 'run_model.py']
UPSTREAM = 'https://github.com/physionetchallenges/python-example-2026.git'

REQUIRED_FILES = ['Dockerfile', 'requirements.txt', 'team_code.py',
                  'train_model.py', 'run_model.py', 'helper_code.py',
                  'channel_table.csv', 'LICENSE', 'AUTHORS.txt', 'README.md']

# Present in training demographics, absent at inference.
TRAINING_ONLY = ['Time_to_Event', 'Last_Known_Visit_Date', 'Time_to_Last_Visit']

results = []


def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f' - {detail}' if detail else ''))
    return ok


def main():
    os.chdir(REPO)
    print('Submission preflight\n' + '=' * 60)

    print('\nRequired files')
    for f in REQUIRED_FILES:
        check(f'{f} present', os.path.exists(f))

    print('\nOrganizer-owned files match upstream')
    ref = '/tmp/pn26_upstream_check'
    git = shutil.which('git') or '/usr/bin/git'
    try:
        if not os.path.exists(git):
            raise FileNotFoundError('git not available on this host')
        if not os.path.isdir(ref):
            subprocess.run([git, 'clone', '--quiet', '--depth', '1', UPSTREAM, ref],
                           check=True, capture_output=True, timeout=180)
        for f in ORGANIZER_OWNED:
            same = subprocess.run(['diff', '-q', f, os.path.join(ref, f)],
                                  capture_output=True).returncode == 0
            check(f'{f} unmodified', same,
                  '' if same else 'differs from upstream; the organizers use theirs')
    except Exception as exc:  # noqa: BLE001
        print(f'  [WARN] upstream comparison skipped: {exc}')
        print('         run this check on a host with git and network access')

    print('\nteam_code imports resolve against requirements.txt')
    declared = {line.split('==')[0].strip().lower()
                for line in open('requirements.txt') if line.strip()}
    alias = {'sklearn': 'scikit-learn'}
    tree = ast.parse(open('team_code.py').read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split('.')[0])
    stdlib = set(sys.stdlib_module_names) | {'helper_code'}
    missing = [m for m in sorted(mods - stdlib)
               if alias.get(m, m).lower() not in declared]
    check('all third-party imports declared', not missing, ', '.join(missing))

    print('\nteam_code uses only helper_code names that exist')
    # The submission was previously unscoreable because team_code called
    # get_standardized_race, which upstream had deleted. `from helper_code
    # import *` hides that until runtime, so resolve every called name against
    # everything team_code actually binds.
    sys.path.insert(0, REPO)
    import builtins as _builtins
    import helper_code

    src = open('team_code.py').read()
    module = ast.parse(src)

    bound = set()
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            bound.update(a.arg for a in getattr(node, 'args', ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[],
                defaults=[])).args)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # Imports inside function bodies bind names too.
            bound.update((a.asname or a.name).split('.')[0] for a in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)

    called = {n.func.id for n in ast.walk(module)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    unknown = sorted(c for c in called
                     if c not in bound
                     and not hasattr(_builtins, c)
                     and not hasattr(helper_code, c))
    check('every called name resolves', not unknown, ', '.join(unknown[:6]))

    print('\nNo training-only columns used as features')
    leaks = [c for c in TRAINING_ONLY if f"'{c}'" in src or f'"{c}"' in src]
    check('no inference-time-absent columns referenced', not leaks, ', '.join(leaks))

    print('\nDocker image contents')
    check('.dockerignore present', os.path.exists('.dockerignore'))
    if os.path.exists('.dockerignore'):
        ignored = {l.strip().rstrip('/') for l in open('.dockerignore')
                   if l.strip() and not l.startswith('#')}
        for d in ['data', 'external', 'saved_models', 'results']:
            check(f'{d}/ excluded from image', d in ignored)

    print('\nNo dangling symlinks that would break COPY')
    dangling = []
    for root, dirs, files in os.walk('.'):
        if any(p in root for p in ('/.git', '/external', '/data', '/node_modules')):
            continue
        for n in dirs + files:
            p = os.path.join(root, n)
            if os.path.islink(p) and not os.path.exists(p):
                dangling.append(p)
    check('no broken symlinks in tracked tree', not dangling, ', '.join(dangling[:3]))

    print('\nModel artifacts are not committed')
    if os.path.exists(git):
        tracked = subprocess.run([git, 'ls-files'], capture_output=True,
                                 text=True).stdout.split()
        big = [f for f in tracked
               if os.path.exists(f) and os.path.getsize(f) > 50 * 2**20]
        check('no tracked file over 50 MB', not big, ', '.join(big[:3]))
    else:
        print('  [WARN] tracked-size check skipped: git unavailable')

    print('\n' + '=' * 60)
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f'{len(failed)} check(s) FAILED:')
        for n in failed:
            print(f'  - {n}')
        return 1
    print(f'All {len(results)} checks passed. Safe to submit.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
