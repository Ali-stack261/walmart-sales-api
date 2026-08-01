"""
Thin entrypoint for FastAPI Cloud's auto-discovery.

FastAPI Cloud's `fastapi run` command only auto-detects an app at specific
default paths (main.py, app.py, api.py, app/main.py, app/app.py, app/api.py).
Our real app lives at api/main.py, which isn't on that list, so this file
re-exports it from the repo root where it will be found.

Local development, Docker, and tests should keep using `api.main:app`
directly — this file exists only for FastAPI Cloud's discovery mechanism.
"""

from api.main import app  # noqa: F401
