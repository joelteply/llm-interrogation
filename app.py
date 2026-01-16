#!/usr/bin/env python3
"""
LLM Interrogator - Web Interface

Extract concepts from LLM training data through statistical probing.
Uses continuation prompts and narrative synthesis for intelligent extraction.

Core cycle: PROBE → VALIDATE → CONDENSE → GROW → repeat
"""

from pathlib import Path
from flask import Flask, render_template

# Create Flask app
app = Flask(
    __name__,
    template_folder="templates_html",
    static_folder="static/dist/assets",
    static_url_path="/assets"
)

# Register blueprints
from routes import projects_bp, probe_bp
from routes.analyze import analyze_bp
from routes.intel import intel_bp
app.register_blueprint(projects_bp)
app.register_blueprint(probe_bp)
app.register_blueprint(analyze_bp)
app.register_blueprint(intel_bp)

# Migrate old single-file projects to new directory structure
from routes.project_storage import migrate_all_old_projects
migrated = migrate_all_old_projects()
if migrated:
    print(f"[STARTUP] Migrated {migrated} projects to new directory structure")

# Start background workers
from workers import start_all_workers
start_all_workers()
print("[STARTUP] Background workers started")


# ============================================================================
# Main routes (serve frontend)
# ============================================================================

@app.route("/")
@app.route("/projects")
@app.route("/project/<path:path>")
def index(path=None):
    """Serve Vite-built frontend."""
    static_index = Path("static/dist/index.html")
    if static_index.exists():
        return static_index.read_text()
    return render_template("app.html")


# Legacy HTML routes (temporary)
@app.route("/old")
def old_index():
    """Old dashboard."""
    from config import TEMPLATES_DIR, load_models_config
    findings = {}
    models = load_models_config()
    templates = [f.stem for f in TEMPLATES_DIR.glob("*.yaml") if not f.stem.startswith("_")]
    return render_template("index.html", findings=findings, models=models, templates=templates)


@app.route("/viewer")
def viewer():
    """Results viewer."""
    from config import RESULTS_DIR
    import json
    results = []
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True)[:50]:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                results.append({
                    "filename": f.name,
                    "timestamp": data.get("timestamp", f.stem),
                    "template": data.get("template", "Unknown"),
                })
            except:
                pass
    return render_template("viewer.html", results=results)


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=5001)
