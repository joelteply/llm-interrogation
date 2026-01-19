"""
Config API routes - serve and update investigation config.
"""

import yaml
from flask import jsonify, request
from pathlib import Path

from . import projects_bp


CONFIG_PATH = Path("templates") / "investigation_config.yaml"


def load_investigation_config() -> dict:
    """Load investigation config from YAML."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_investigation_config(config: dict) -> None:
    """Save investigation config to YAML."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


@projects_bp.route("/api/config/investigation", methods=["GET"])
def get_investigation_config():
    """Get the investigation config."""
    config = load_investigation_config()
    return jsonify(config)


@projects_bp.route("/api/config/investigation", methods=["PUT"])
def update_investigation_config():
    """Update the investigation config."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Merge with existing config
    config = load_investigation_config()
    config.update(data)

    save_investigation_config(config)
    return jsonify({"status": "saved", "config": config})


@projects_bp.route("/api/config/investigation/examples", methods=["PUT"])
def update_examples():
    """Update just the examples list."""
    data = request.json
    examples = data.get("examples", [])

    config = load_investigation_config()
    config["examples"] = examples

    save_investigation_config(config)
    return jsonify({"status": "saved", "examples": examples})


@projects_bp.route("/api/config/investigation/goals", methods=["PUT"])
def update_goals():
    """Update goal detection keywords."""
    data = request.json
    goals = data.get("goals", {})

    config = load_investigation_config()
    config["goals"] = goals

    save_investigation_config(config)
    return jsonify({"status": "saved", "goals": goals})
