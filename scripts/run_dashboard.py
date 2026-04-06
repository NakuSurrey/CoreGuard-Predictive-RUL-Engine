"""
Dashboard Startup Script — CoreGuard Predictive RUL Engine.

Starts the Streamlit dashboard with a single command:

    python scripts/run_dashboard.py

IMPORTANT: The API server must be running FIRST.
Start the API with:  python scripts/serve.py

After starting, the dashboard is available at:
    http://localhost:8501
"""

import sys
import os
import subprocess

# adding project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    """Start the Streamlit dashboard."""

    # path to the dashboard app file
    app_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")
    app_path = os.path.abspath(app_path)

    if not os.path.exists(app_path):
        print(f"[dashboard] ERROR: Dashboard file not found at {app_path}")
        sys.exit(1)

    print("=" * 60)
    print("CoreGuard Predictive RUL Engine — Dashboard")
    print("=" * 60)
    print(f"  Dashboard: http://localhost:8501")
    print(f"  API must be running at: http://localhost:8000")
    print("=" * 60)

    # launch streamlit as a subprocess
    # --server.headless=true prevents streamlit from trying to open a browser
    # --server.port=8501 sets the dashboard port
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ])


if __name__ == "__main__":
    main()
