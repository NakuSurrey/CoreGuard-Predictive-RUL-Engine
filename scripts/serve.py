"""
API Server Startup Script — CoreGuard Predictive RUL Engine.

Starts the FastAPI server with a single command:

    python scripts/serve.py

This runs uvicorn (the ASGI server) which loads src/api/main.py
and starts listening for HTTP requests.

After starting, the API is available at:
    http://localhost:8000

Interactive API docs (Swagger UI) are at:
    http://localhost:8000/docs

Alternative docs (ReDoc) are at:
    http://localhost:8000/redoc
"""

import sys
import os

# adding project root to path so imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
from src.config import API_HOST, API_PORT


def main():
    """Start the API server."""
    print("=" * 60)
    print("CoreGuard Predictive RUL Engine — API Server")
    print("=" * 60)
    print(f"  Host: {API_HOST}")
    print(f"  Port: {API_PORT}")
    print(f"  Docs: http://localhost:{API_PORT}/docs")
    print("=" * 60)

    # uvicorn.run() starts the web server
    #   "src.api.main:app" tells uvicorn where to find the FastAPI app object
    #   reload=True means the server restarts automatically when code changes
    #   (useful during development, would be False in production)
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
