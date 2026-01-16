"""
Intelligence Extraction System

Extracts private information from LLM training data.
Distinguishes real leaks from hallucinations using behavioral testing.
"""

from flask import Blueprint

intel_bp = Blueprint("intel", __name__)

from . import api  # noqa: E402, F401
