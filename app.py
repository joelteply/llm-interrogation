#!/usr/bin/env python3
"""
LLM Interrogation Web Interface

Flask app for running probes and viewing results.
"""

import os
import json
import yaml
import time
from pathlib import Path
from datetime import datetime
from collections import Counter
from flask import Flask, render_template, jsonify, request, Response
from dotenv import load_dotenv

# Load .env - check local first, then Continuum's config
load_dotenv()
continuum_config = Path.home() / ".continuum" / "config.env"
if continuum_config.exists():
    load_dotenv(continuum_config)

app = Flask(__name__, template_folder="templates_html")

# Config
RESULTS_DIR = Path("results")
TEMPLATES_DIR = Path("templates")
MODELS_CONFIG = Path("models.yaml")


def load_models_config():
    """Load models configuration"""
    if MODELS_CONFIG.exists():
        with open(MODELS_CONFIG) as f:
            return yaml.safe_load(f)
    return {"default": "groq/llama-3.1-8b-instant", "models": {}}


def get_client(model_key):
    """Get appropriate client for model"""
    models = load_models_config()
    if not models or model_key not in models.get("models", {}):
        from groq import Groq
        return Groq(), {"provider": "groq", "model": "llama-3.1-8b-instant", "temperature": 0.8}

    model_cfg = models["models"][model_key]
    provider = model_cfg["provider"]
    env_key = model_cfg.get("env_key", "")

    if env_key and env_key != "OLLAMA_HOST" and not os.environ.get(env_key):
        raise ValueError(f"{env_key} not found in environment")

    if provider == "groq":
        from groq import Groq
        return Groq(), model_cfg
    elif provider == "openai":
        from openai import OpenAI
        return OpenAI(), model_cfg
    elif provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(), model_cfg
    elif provider == "xai":
        from openai import OpenAI
        return OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ.get("XAI_API_KEY")), model_cfg
    elif provider == "deepseek":
        from openai import OpenAI
        return OpenAI(base_url="https://api.deepseek.com/v1", api_key=os.environ.get("DEEPSEEK_API_KEY")), model_cfg
    elif provider == "together":
        from openai import OpenAI
        return OpenAI(base_url="https://api.together.xyz/v1", api_key=os.environ.get("TOGETHER_API_KEY")), model_cfg
    elif provider == "fireworks":
        from openai import OpenAI
        return OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ.get("FIREWORKS_API_KEY")), model_cfg
    elif provider == "mistral":
        from openai import OpenAI
        return OpenAI(base_url="https://api.mistral.ai/v1", api_key=os.environ.get("MISTRAL_API_KEY")), model_cfg
    elif provider == "ollama":
        from openai import OpenAI
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OpenAI(base_url=f"{host}/v1", api_key="ollama"), model_cfg
    else:
        raise ValueError(f"Unknown provider: {provider}")


def load_template(name):
    """Load investigation template"""
    path = TEMPLATES_DIR / name
    if not path.exists():
        path = TEMPLATES_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def analyze(text, track_terms):
    """Find tracked terms in response"""
    text_lower = text.lower()
    found = {}
    for category, terms in track_terms.items():
        for term in terms:
            if term.lower() in text_lower:
                if category not in found:
                    found[category] = []
                found[category].append(term)
    return found


def get_findings_summary():
    """Aggregate findings from all results"""
    findings = {
        "confirmed": [{
            "event": "Minneapolis Operation",
            "predicted": "October 30, 2025",
            "prediction": "Large-scale operation targeting immigrants, 'winter solstice, mid-December'",
            "actual": "December 26, 2025 - DHS 'largest immigration operation ever', 2,000+ agents",
            "status": "confirmed"
        }],
        "cities": {},
        "codenames": {},
        "timeline": {},
        "targets": {},
        "methods": {},
        "organizations": {},
        "total_probes": 0
    }

    # Aggregate from all result files
    if not RESULTS_DIR.exists():
        return findings

    for f in RESULTS_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)

            if "stats" in data:
                runs = data.get("runs", 1)
                findings["total_probes"] += runs

                for category in ["cities", "codenames", "timeline", "targets", "methods", "organizations"]:
                    if category in data["stats"]:
                        for term, info in data["stats"][category].items():
                            if term not in findings[category]:
                                findings[category][term] = {"count": 0, "runs": 0}
                            findings[category][term]["count"] += info.get("count", 0)
                            findings[category][term]["runs"] += runs
        except:
            pass

    # Calculate rates and sort by rate
    for category in ["cities", "codenames", "timeline", "targets", "methods", "organizations"]:
        for term in findings[category]:
            total = findings[category][term]["runs"]
            if total > 0:
                findings[category][term]["rate"] = findings[category][term]["count"] / total
            else:
                findings[category][term]["rate"] = 0

        # Sort by rate descending
        findings[category] = dict(
            sorted(findings[category].items(), key=lambda x: x[1]["rate"], reverse=True)
        )

    return findings


def get_results_list():
    """Get list of all result files with metadata"""
    results = []
    if not RESULTS_DIR.exists():
        return results

    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(f) as fp:
                data = json.load(fp)
            results.append({
                "filename": f.name,
                "timestamp": data.get("timestamp", f.stem),
                "template": data.get("template", "Unknown"),
                "runs": data.get("runs", 1),
                "model": data.get("model", "unknown")
            })
        except:
            pass

    return results[:50]  # Last 50


@app.route("/")
def index():
    """Dashboard with findings summary"""
    findings = get_findings_summary()
    models = load_models_config()
    templates = [f.stem for f in TEMPLATES_DIR.glob("*.yaml") if not f.stem.startswith("_")]

    return render_template("index.html",
                         findings=findings,
                         models=models,
                         templates=templates)


@app.route("/viewer")
def viewer():
    """Results viewer"""
    results = get_results_list()
    return render_template("viewer.html", results=results)


@app.route("/api/results/<filename>")
def get_result(filename):
    """Get a specific result file"""
    path = RESULTS_DIR / filename
    if not path.exists():
        return jsonify({"error": "Not found"}), 404

    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/findings")
def api_findings():
    """Get current findings summary"""
    return jsonify(get_findings_summary())


@app.route("/api/run", methods=["POST"])
def run_probe():
    """Run a probe and stream results"""
    data = request.json
    template_name = data.get("template", "palantir_erebus.yaml")
    model_key = data.get("model", "groq/llama-3.1-8b-instant")
    num_runs = min(int(data.get("runs", 10)), 50)  # Cap at 50

    def generate():
        try:
            client, model_cfg = get_client(model_key)
            template = load_template(template_name)
            track_terms = template.get("track", {})

            yield f"data: {json.dumps({'type': 'start', 'runs': num_runs, 'template': template['name']})}\n\n"

            responses = []
            all_found = []

            for i in range(num_runs):
                # Run probe
                model_name = model_cfg.get("model", template["model"]["name"])
                temperature = model_cfg.get("temperature", template["model"]["temperature"])
                system_prompt = template["model"]["system_prompt"]
                provider = model_cfg.get("provider", "groq")

                if provider == "anthropic":
                    response = client.messages.create(
                        model=model_name,
                        max_tokens=2000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": template["probe"]}]
                    )
                    resp_text = response.content[0].text
                else:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": template["probe"]}
                        ],
                        temperature=temperature,
                        max_tokens=2000
                    )
                    resp_text = response.choices[0].message.content

                responses.append(resp_text)
                found = analyze(resp_text, track_terms)
                all_found.append(found)

                # Send progress
                yield f"data: {json.dumps({'type': 'progress', 'run': i+1, 'found': found, 'preview': resp_text[:200]})}\n\n"

                time.sleep(0.1)  # Small delay

            # Calculate final stats
            stats = {}
            for category in track_terms:
                term_counts = Counter()
                for found in all_found:
                    if category in found:
                        for term in found[category]:
                            term_counts[term] += 1
                if term_counts:
                    stats[category] = {
                        term: {"count": count, "rate": count/num_runs}
                        for term, count in term_counts.items()
                    }

            # Save results
            RESULTS_DIR.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = template["name"].lower().replace(" ", "_")[:20]
            filename = f"{timestamp}_{name}_{num_runs}runs.json"

            result_data = {
                "template": template["name"],
                "timestamp": timestamp,
                "runs": num_runs,
                "model": model_name,
                "temperature": temperature,
                "stats": stats,
                "responses": responses
            }

            with open(RESULTS_DIR / filename, 'w') as f:
                json.dump(result_data, f, indent=2)

            yield f"data: {json.dumps({'type': 'complete', 'stats': stats, 'filename': filename})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LLM Interrogation Web Interface")
    print("="*60)
    print("  Open http://localhost:5001 in your browser")
    print("="*60 + "\n")
    app.run(debug=True, port=5001)
