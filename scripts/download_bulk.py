#!/usr/bin/env python
"""Bulk-download a Challenge dataset as a single archive, with HTTP resume.

Kaggle rate-limits the per-file download endpoint hard: fetching 3,292 files
individually returns 429 after roughly 1,600, no matter how the requests are
paced. The bulk endpoint is one request for the whole dataset, so it sidesteps
the limit entirely.

The stock `dataset_download_files` has no resume, which is unacceptable for a
1.2 TiB stream. This issues the request directly with a Range header so an
interrupted transfer picks up where it stopped, then extracts only the members
that are missing or the wrong size.

Usage:
    python scripts/download_bulk.py --dataset physionet/physionetchallenge2026data \
        --dest data/raw/training_set_small --workdir data/archives
"""

import argparse
import os
import sys
import time
import zipfile

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

CHUNK = 1 << 23  # 8 MiB


def archive_url(dataset):
    owner, name = dataset.split('/', 1)
    return f'https://www.kaggle.com/api/v1/datasets/download/{owner}/{name}'


def fetch(dataset, archive_path, credentials, max_attempts=40):
    """Stream the archive to disk, resuming across interruptions."""
    url = archive_url(dataset)
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        have = os.path.getsize(archive_path) if os.path.exists(archive_path) else 0
        headers = {'Range': f'bytes={have}-'} if have else {}

        try:
            with requests.get(url, auth=credentials, headers=headers,
                              stream=True, timeout=(30, 300),
                              allow_redirects=True) as resp:
                if resp.status_code == 416:
                    print('Server reports the range is satisfied; archive complete.',
                          flush=True)
                    return True
                if resp.status_code == 429:
                    wait = min(600, 30 * attempt)
                    print(f'429 throttled; sleeping {wait}s', flush=True)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()

                # A 200 to a Range request means the server ignored it, so the
                # body starts from zero and any partial file must be discarded.
                if have and resp.status_code == 200:
                    print('Server ignored Range; restarting from byte 0.', flush=True)
                    have = 0
                    mode = 'wb'
                else:
                    mode = 'ab' if have else 'wb'

                total = resp.headers.get('Content-Length')
                total = int(total) + have if total else None
                start, last_report = time.time(), have

                with open(archive_path, mode) as fh:
                    for chunk in resp.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        have += len(chunk)
                        if have - last_report >= (1 << 30):
                            rate = (have - last_report) / (time.time() - start + 1e-9)
                            pct = f'{100 * have / total:.1f}%' if total else '?'
                            print(f'  {have / 2**30:8.1f} GiB  {pct}  '
                                  f'{rate / 2**20:.0f} MiB/s', flush=True)
                            last_report, start = have, time.time()

            if total is None or have >= total:
                print(f'Archive complete: {have / 2**30:.1f} GiB', flush=True)
                return True
            print(f'Short read ({have}/{total}); resuming', flush=True)

        except (requests.RequestException, OSError) as exc:
            wait = min(300, 15 * attempt)
            print(f'attempt {attempt} failed ({type(exc).__name__}: {exc}); '
                  f'retrying in {wait}s', flush=True)
            time.sleep(wait)

    return False


def extract(archive_path, dest):
    """Extract members that are missing or the wrong size."""
    os.makedirs(dest, exist_ok=True)
    written = skipped = 0

    with zipfile.ZipFile(archive_path) as zf:
        members = zf.infolist()
        print(f'Archive holds {len(members)} members', flush=True)
        for i, info in enumerate(members, 1):
            if info.is_dir():
                continue
            target = os.path.join(dest, info.filename)
            if os.path.exists(target) and os.path.getsize(target) == info.file_size:
                skipped += 1
                continue
            os.makedirs(os.path.dirname(target) or dest, exist_ok=True)
            with zf.open(info) as src, open(target, 'wb') as out:
                while True:
                    block = src.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
            written += 1
            if i % 200 == 0:
                print(f'  extracted {i}/{len(members)} '
                      f'(new={written} skipped={skipped})', flush=True)

    print(f'Extraction done: {written} written, {skipped} already present',
          flush=True)
    return written, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dest', required=True)
    parser.add_argument('--workdir', default='data/archives')
    parser.add_argument('--keep-archive', action='store_true',
                        help='retain the zip after extraction')
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()
    credentials = (api.config_values['username'], api.config_values['key'])

    os.makedirs(args.workdir, exist_ok=True)
    archive = os.path.join(args.workdir,
                           args.dataset.split('/')[-1] + '.zip')

    print(f'dataset={args.dataset}\narchive={archive}\ndest={args.dest}', flush=True)

    if not fetch(args.dataset, archive, credentials):
        print('Download did not complete.', flush=True)
        return 1

    if not zipfile.is_zipfile(archive):
        print('Downloaded file is not a valid zip; removing so the next run '
              'starts clean.', flush=True)
        os.remove(archive)
        return 1

    extract(archive, args.dest)

    if not args.keep_archive:
        os.remove(archive)
        print('Removed archive.', flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
