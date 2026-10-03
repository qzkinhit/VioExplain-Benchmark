"""Download fixed official dataset snapshots; do not execute upstream code.

Git datasets use a fixed data subtree and retained notices. TEP uses the
official archive with a fixed SHA256, including its own readme. Raw files
remain local and ignored; per-file SHA256 hashes are recorded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile


def run(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['smd','skab','tep'])
    parser.add_argument('--out', type=Path, default=Path('data/raw'))
    parser.add_argument('--show-source', action='store_true', help='print pinned source without downloading')
    args=parser.parse_args()
    registry=json.loads(Path(__file__).with_name('sources.lock.json').read_text())
    info=registry['datasets'][args.dataset]
    if args.show_source:
        print(json.dumps(info,indent=2)); return
    target=args.out/args.dataset
    if target.exists():
        raise SystemExit(f'Refusing to replace existing data directory: {target}')
    args.out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='vioexplain-download-',dir=args.out) as tmp:
        staging=Path(tmp)/'dataset'
        if info.get('kind') == 'zip':
            archive=Path(tmp)/'upstream.zip'
            urllib.request.urlretrieve(info['url'],archive)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != info['sha256']:
                raise RuntimeError('Downloaded archive SHA256 does not match the lock')
            checkout=Path(tmp)/'upstream';checkout.mkdir()
            with zipfile.ZipFile(archive) as z:
                for item in z.infolist():
                    path=Path(item.filename)
                    if path.is_absolute() or '..' in path.parts or ((item.external_attr >> 16) & 0o170000) == 0o120000:
                        raise RuntimeError('Refusing unsafe archive member')
                z.extractall(checkout)
            shutil.copytree(checkout/info['subdirectory'],staging)
            revision=info['sha256']
        else:
            checkout=Path(tmp)/'upstream'
            run('git','clone','--filter=blob:none','--no-checkout',info['url'],str(checkout))
            run('git','sparse-checkout','init','--cone',cwd=checkout)
            run('git','sparse-checkout','set',info['subdirectory'],cwd=checkout)
            run('git','checkout','--detach',info['revision'],cwd=checkout)
            revision=run('git','rev-parse','HEAD',cwd=checkout)
            if revision != info['revision']:
                raise RuntimeError('Downloaded revision does not match the lock')
            shutil.copytree(checkout/info['subdirectory'],staging)
            notices=staging/'UPSTREAM_NOTICES';notices.mkdir(exist_ok=True)
            for name in ['LICENSE','LICENSE.md','LICENSE.txt','README.md']:
                item=checkout/name
                if item.is_file():shutil.copyfile(item,notices/name)
        hashes={}
        for path in sorted(staging.rglob('*')):
            if path.is_file():
                if path.is_symlink():raise RuntimeError('Refusing symlink in data snapshot')
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
                hashes[path.relative_to(staging).as_posix()]=h.hexdigest()
        manifest={'schema_version':1,'dataset':args.dataset,**info,'file_sha256':hashes,
                  'executed_upstream_code':False,'redistributed_with_source_release':False}
        (staging/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        staging.rename(target)
    print(f'Downloaded {args.dataset} at {revision}; manifest: {target / "download_manifest.json"}')


if __name__=='__main__':main()
