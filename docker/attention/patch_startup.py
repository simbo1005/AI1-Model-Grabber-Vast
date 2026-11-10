"""Fail image build if provider's real launch command no longer matches."""
import re
import sys
from pathlib import Path
family = sys.argv[1]
if family == 'vast':
    path = Path('/opt/supervisor-scripts/comfyui.sh')
    pattern = r'pty python main[.]py'
    replacement = 'pty python /opt/dsnn-attention/launch.py --base-directory "${COMFYUI_DIR}"'
else:
    path = Path('/usr/local/bin/runpod-base-start.sh')
    pattern = r'(?m)^([ \t]*)python[ \t]+main[.]py(?=[ \t]|$)'
    replacement = r'\1python /opt/dsnn-attention/launch.py --base-directory "$COMFYUI_DIR"'
source, count = re.subn(pattern, replacement, path.read_text())
if count != 1:
    raise RuntimeError('Provider startup changed: expected exactly one ComfyUI launch command')
path.write_text(source)
