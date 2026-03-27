from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def launch_dashboard() -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is not installed. Run `pip install -e \".[dashboard]\"` first."
        ) from exc

    app_path = Path(__file__).with_name("dashboard_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))
