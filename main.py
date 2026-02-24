"""
Graph Extraction Pipeline — thin wrapper around the knwler package.

All functionality has been moved into the ``knwler`` package.
This file is kept for backwards compatibility so that
``python main.py`` continues to work.
"""

from knwler.cli import app

if __name__ == "__main__":
    app()
