#!/usr/bin/env python3
"""Freeze a completed local qualification bundle and source tree with checksums.

This only writes local files. Existing archives are never overwritten. Run after
all measurement processes finish, with PYTHONDONTWRITEBYTECODE=1.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('target/qualification'))
    p.add_argument('--archive', type=Path, default=Path('benchmarks/artifacts/core-after'))
    p.add_argument('--manifest', type=Path, default=Path('benchmarks/baselines/core-after.json'))
    a = p.parse_args()
    os.chdir(ROOT)
    source, archive, manifest = a.source.resolve(), a.archive.resolve(), a.manifest.resolve()
    checksums = manifest.with_name(manifest.stem + '-artifacts.json')
    if archive.exists() or manifest.exists() or checksums.exists():
        raise SystemExit('Choose unused archive/manifest paths; existing evidence is never overwritten.')
    if not (source / 'summary.json').is_file():
        raise SystemExit('Assemble the report before archiving it.')
    if source == archive or source in archive.parents or archive in source.parents:
        raise SystemExit('Source and archive must be separate directory trees.')
    files = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard']).decode().split('\0')
    files = sorted({f for f in files if f and Path(f).is_file() and Path(f).resolve() not in (manifest, checksums)})
    source_hashes = {f: digest(Path(f)) for f in files}
    archive.parent.mkdir(parents=True, exist_ok=True)
    if platform.system() == 'Darwin':
        # APFS clone: preserve large audio/binary evidence without duplicating
        # its physical blocks. Subsequent writes cannot alter the original.
        subprocess.run(['cp', '-cR', str(source), str(archive)], check=True)
    else:
        shutil.copytree(source, archive)
    with tarfile.open(archive / 'sources.tar.gz', 'w:gz') as tar:
        for f in files:
            tar.add(f, arcname=f, recursive=False)
    artifacts = {}
    for path in sorted(archive.rglob('*')):
        if path.is_file():
            artifacts[str(path.relative_to(archive))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    write(checksums, {'archive': str(archive.relative_to(ROOT)), 'files': artifacts})
    write(manifest, {
        'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'archive': str(archive.relative_to(ROOT)),
        'artifact_checksums': str(checksums.relative_to(ROOT)),
        'artifact_checksums_sha256': digest(checksums),
        'artifact_count': len(artifacts),
        'source_archive_sha256': artifacts['sources.tar.gz']['sha256'],
        'source_sha256': source_hashes,
        'qualification_summary': json.loads((archive / 'summary.json').read_text()),
        'scope': 'Local evidence only; preserves failed gates and per-layer source/binary provenance. No publishing.'
    })
    print(f'Archived {len(artifacts)} artifacts and {len(files)} source files to {archive}')
    print(f'Manifest: {manifest}')


if __name__ == '__main__':
    main()
