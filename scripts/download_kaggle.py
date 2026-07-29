#!/usr/bin/env python
"""Resumable per-file downloader for the Challenge 2026 Kaggle datasets.

The official datasets are 214 GiB (small) and 1.2 TiB (large). A single archive
download is fragile at that size, so we enumerate the dataset file-by-file and
fetch anything not already on disk. Re-running the script picks up where it left
off; already-complete files are skipped by size comparison.

Usage:
    python scripts/download_kaggle.py --dataset physionet/physionetchallenge2026data \
        --dest data/raw/training_set_small
"""

import argparse
import os
import sys
import threading
import time

from kaggle.api.kaggle_api_extended import KaggleApi


def list_dataset_files(api, dataset):
    """Enumerate every file in a dataset, following pagination."""
    files = []
    token = None
    while True:
        page = api.dataset_list_files(dataset, page_token=token, page_size=1000)
        if getattr(page, 'error_message', None):
            raise RuntimeError(page.error_message)
        batch = list(page.files)
        files.extend(
            (f.name, int(f.total_bytes) if f.total_bytes is not None else None)
            for f in batch
        )
        token = getattr(page, 'next_page_token', None)
        if not token or not batch:
            break
    return files


def already_complete(path, expected_size):
    if not os.path.exists(path):
        return False
    if expected_size is None:
        return True
    return os.path.getsize(path) == expected_size


class RateLimiter:
    """Serialize request starts with a minimum gap between them.

    Kaggle throttles the per-file download endpoint under concurrency: at six
    workers every request began returning 404 after roughly 1,600 files, while a
    single request still succeeded immediately. Pacing request starts keeps the
    sweep alive over the hours a 214 GiB pull takes.
    """

    def __init__(self, min_interval):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.min_interval
        if sleep_for:
            time.sleep(sleep_for)


def download_one(api, dataset, remote_name, dest_root, expected_size, retries=5,
                 limiter=None):
    owner, name = dataset.split('/', 1)
    target = os.path.join(dest_root, remote_name)

    if already_complete(target, expected_size):
        return 'skip'

    os.makedirs(os.path.dirname(target), exist_ok=True)

    for attempt in range(retries):
        try:
            if limiter is not None:
                limiter.wait()
            # dataset_download_file writes into `path` preserving the remote
            # subdirectory layout, and unzips single-file archives itself.
            api.dataset_download_file(
                f'{owner}/{name}',
                file_name=remote_name,
                path=os.path.dirname(target),
                force=False,
                quiet=True,
            )
            # Kaggle sometimes lands the payload as `<name>.zip`; unpack it.
            zipped = target + '.zip'
            if os.path.exists(zipped) and not os.path.exists(target):
                import zipfile
                with zipfile.ZipFile(zipped) as zf:
                    zf.extractall(os.path.dirname(target))
                os.remove(zipped)
            if already_complete(target, expected_size):
                return 'ok'
            raise RuntimeError(f'size mismatch after download: {remote_name}')
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            if attempt == retries - 1:
                print(f'FAILED {remote_name}: {exc}', flush=True)
                return 'fail'
            # Exponential backoff. A 404 here means throttling, not a missing
            # file, so backing off hard is what actually recovers the sweep.
            time.sleep(min(120, 5 * (2 ** attempt)))
    return 'fail'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='owner/dataset-slug')
    parser.add_argument('--dest', required=True, help='destination directory')
    parser.add_argument('--workers', type=int, default=2,
                        help='concurrent downloads; Kaggle throttles above ~2')
    parser.add_argument('--min-interval', type=float, default=0.35,
                        help='minimum seconds between request starts')
    parser.add_argument('--manifest', default=None,
                        help='cache the file listing here to avoid re-enumerating')
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()

    if args.manifest and os.path.exists(args.manifest):
        with open(args.manifest) as fh:
            files = []
            for line in fh:
                name, size = line.rstrip('\n').rsplit('\t', 1)
                files.append((name, int(size) if size != 'None' else None))
        print(f'Loaded {len(files)} files from manifest {args.manifest}', flush=True)
    else:
        print(f'Enumerating {args.dataset} ...', flush=True)
        files = list_dataset_files(api, args.dataset)
        print(f'Found {len(files)} files', flush=True)
        if args.manifest:
            os.makedirs(os.path.dirname(args.manifest) or '.', exist_ok=True)
            with open(args.manifest, 'w') as fh:
                for name, size in files:
                    fh.write(f'{name}\t{size}\n')

    total_bytes = sum(s for _, s in files if s)
    todo = [(n, sz) for n, sz in files
            if not already_complete(os.path.join(args.dest, n), sz)]
    todo_bytes = sum(sz for _, sz in todo if sz)
    print(f'Total {total_bytes / 2**30:.1f} GiB across {len(files)} files; '
          f'{len(todo)} still needed ({todo_bytes / 2**30:.1f} GiB)', flush=True)
    files = todo

    os.makedirs(args.dest, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    limiter = RateLimiter(args.min_interval)
    counts = {'ok': 0, 'skip': 0, 'fail': 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, api, args.dataset, name, args.dest, size,
                        limiter=limiter): name
            for name, size in files
        }
        for i, fut in enumerate(as_completed(futures), 1):
            counts[fut.result()] += 1
            if i % 25 == 0 or i == len(files):
                print(f'{i}/{len(files)}  ok={counts["ok"]} '
                      f'skip={counts["skip"]} fail={counts["fail"]}', flush=True)

    print(f'Done: {counts}', flush=True)
    return 1 if counts['fail'] else 0


if __name__ == '__main__':
    sys.exit(main())
