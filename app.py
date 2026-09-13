"""
Root entry point for Streamlit Community Cloud (and `streamlit run app.py` locally).
All page logic lives in dashboard/app.py; this file only makes it importable as the
top-level script and keeps the deployment's "main file path" at the repository root.
"""
from dashboard import app  # noqa: F401  (executes the dashboard on import)
