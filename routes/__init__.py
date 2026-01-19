"""
Flask blueprints for the LLM Interrogator API.
"""

from flask import Blueprint

# Create blueprints
projects_bp = Blueprint('projects', __name__)
probe_bp = Blueprint('probe', __name__)
legacy_bp = Blueprint('legacy', __name__)
# Import routes to register them
from . import projects
from . import probe
from . import generate
from . import synthesize
from . import config_api
