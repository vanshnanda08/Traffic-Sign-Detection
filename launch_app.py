"""Portable, explicit launcher; never installs packages or downloads weights."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--offline-sandbox', action='store_true', help='macOS only: deny external network for server and all children; allow loopback.')
    args = parser.parse_args()
    if not 1024<=args.port<=65535:
        parser.error('Choose an unprivileged port between 1024 and 65535.')
    suffix = 'Scripts/python.exe' if os.name=='nt' else 'bin/python'
    ui = ROOT/'.cache/app-venv'/suffix
    sci = Path(os.environ.get('P57_INFERENCE_PYTHON', ROOT/'.venv'/suffix))
    if not ui.is_file() or not sci.is_file():
        parser.error('Missing UI or scientific environment. Follow app/README.md installation; no automatic install is performed.')
    checked = subprocess.run([str(ui), str(ROOT/'app/check_ui_environment.py')], capture_output=True, text=True)
    if checked.returncode:
        parser.error('Missing/incompatible UI dependencies. Sync app/requirements-ui.txt in .cache/app-venv.')
    env = {**os.environ, 'STREAMLIT_BROWSER_GATHER_USAGE_STATS':'false',
           'PYTHONDONTWRITEBYTECODE':'1'}
    cmd = [str(ui), '-m', 'streamlit', 'run', str(ROOT/'app/streamlit_app.py'),
           '--server.address=127.0.0.1', f'--server.port={args.port}',
           '--browser.gatherUsageStats=false', '--server.headless=true']
    if args.offline_sandbox:
        if sys.platform!='darwin' or not Path('/usr/bin/sandbox-exec').exists():
            parser.error('This offline sandbox verification option requires macOS sandbox-exec; no silent unblocked launch.')
        profile = '(version 1)(allow default)(deny network*)(allow network-outbound (remote ip "localhost:*"))(allow network-inbound (local ip "localhost:*"))'
        cmd = ['/usr/bin/sandbox-exec', '-p', profile, *cmd]
    print(f'P57 local app: http://127.0.0.1:{args.port} (Ctrl+C to stop)', flush=True)
    try:
        return subprocess.call(cmd, cwd=ROOT/'app', env=env)
    except KeyboardInterrupt:
        return 130


if __name__=='__main__':
    raise SystemExit(main())
