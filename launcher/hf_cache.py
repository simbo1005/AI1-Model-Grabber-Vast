"""Seed a Transformers snapshot from verified CLI downloads; no network transfer."""
import json
import os
from pathlib import Path
import re
import sys

def seed(root, specs, cache):
    root = Path(root).resolve()
    for spec in specs:
        repo, revision = spec['repo'], spec['revision']
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo) or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Invalid cache snapshot identity')
        directory = (root/spec['directory']).resolve()
        if not directory.is_relative_to(root/'models'):
            raise ValueError('Cache source must be a model directory')
        sources = [(name, (directory/name).resolve()) for name in spec['files']]
        if any(not path.is_relative_to(directory) or not path.is_file() for _, path in sources):
            raise RuntimeError('HF cache snapshot is incomplete: '+repo)
        store = Path(cache)/('models--'+repo.replace('/', '--'))
        snapshot = store/'snapshots'/revision
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, source in sources:
            target = snapshot/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and not target.is_symlink():
                continue  # Keep HF's existing real cache files.
            temp = target.with_name(target.name+'.dsnn-part')
            temp.unlink(missing_ok=True)
            temp.symlink_to(source)
            os.replace(temp, target)
        refs = store/'refs'
        refs.mkdir(exist_ok=True)
        (refs/'main').write_text(revision)

if __name__ == '__main__':
    from huggingface_hub.constants import HF_HUB_CACHE
    seed(sys.argv[1], json.loads(sys.argv[2]), HF_HUB_CACHE)
