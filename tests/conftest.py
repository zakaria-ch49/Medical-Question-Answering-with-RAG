"""
pytest conftest.py
==================
Shared fixtures and path configuration for the test suite.

Adds the ``src/`` directory to ``sys.path`` so every test file can import
project modules (download_pubmed, open_router, bio_clinical_embeddings …)
without repeating the path manipulation.
"""
import sys
import os

# Make all src/ modules importable by default
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
