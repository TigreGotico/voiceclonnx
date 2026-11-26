import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_quants_page_equals_the_committed_results():
    done = subprocess.run([sys.executable, "scripts/render_quants.py", "--check"], cwd=ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout
