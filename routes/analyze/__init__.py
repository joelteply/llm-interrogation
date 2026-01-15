"""
Document Analysis Package - Unredaction, Enrichment, Discovery

Modular pipeline for extracting private knowledge from LLMs using document seeds.
"""

from flask import Blueprint

analyze_bp = Blueprint('analyze', __name__)

# Import routes after blueprint creation to avoid circular imports
from . import api  # noqa: E402, F401
