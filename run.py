#!/usr/bin/env python3
"""
LLM Interrogation Runner

Simple interface to run investigations from templates.

Usage:
    python run.py                     # Run default (Palantir/Erebus) with default model
    python run.py --runs 20           # Run 20 times for better stats
    python run.py -t mytemplate.yaml  # Use custom template
    python run.py -m openai/gpt-4o    # Use different model
    python run.py --models            # List available models
    python run.py --list              # List available templates
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv

# Load .env - check local first, then Continuum's config
load_dotenv()  # Local .env
continuum_config = Path.home() / ".continuum" / "config.env"
if continuum_config.exists():
    load_dotenv(continuum_config)  # Fallback to Continuum's keys

DEFAULT_TEMPLATE = "templates/palantir_erebus.yaml"
MODELS_CONFIG = "models.yaml"


def load_models_config():
    """Load models configuration"""
    if os.path.exists(MODELS_CONFIG):
        with open(MODELS_CONFIG) as f:
            return yaml.safe_load(f)
    return None


def get_client(model_key):
    """Get appropriate client for model"""
    models = load_models_config()
    if not models or model_key not in models.get("models", {}):
        # Fallback to Groq
        try:
            from groq import Groq
            return Groq(), {"provider": "groq", "model": "llama-3.1-8b-instant", "temperature": 0.8}
        except:
            print("No model configuration found and groq not available")
            sys.exit(1)

    model_cfg = models["models"][model_key]
    provider = model_cfg["provider"]
    env_key = model_cfg.get("env_key", "")

    # Check API key
    if env_key and env_key != "OLLAMA_HOST" and not os.environ.get(env_key):
        print(f"{env_key} not found in .env!")
        print(f"Add it to .env file to use {model_key}")
        sys.exit(1)

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
        print(f"Unknown provider: {provider}")
        sys.exit(1)


# Global client - set by main()
client = None
model_config = None


def load_template(path):
    """Load investigation template"""
    with open(path) as f:
        return yaml.safe_load(f)


def probe(prompt, config):
    """Send probe to model"""
    global client, model_config

    # Use model from config or global model_config
    model_name = model_config.get("model", config["model"]["name"])
    temperature = model_config.get("temperature", config["model"]["temperature"])
    system_prompt = config["model"]["system_prompt"]
    provider = model_config.get("provider", "groq")

    if provider == "anthropic":
        # Anthropic has different API
        response = client.messages.create(
            model=model_name,
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    else:
        # OpenAI-compatible API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=2000
        )
        return response.choices[0].message.content


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


def print_header(config):
    """Print investigation header"""
    global model_config
    print("\n" + "=" * 60)
    print(f"  {config['name']}")
    print("=" * 60)
    print(f"  {config['description']}")
    model_name = model_config.get("model", config['model']['name'])
    provider = model_config.get("provider", "groq")
    temp = model_config.get("temperature", config['model']['temperature'])
    print(f"  Model: {provider}/{model_name} @ temp {temp}")
    print("=" * 60)


def print_findings(all_found, num_runs, track_terms):
    """Print findings summary with corroboration levels"""
    print("\n" + "=" * 60)
    print("  FINDINGS - Corroboration Across {} Runs".format(num_runs))
    print("=" * 60)
    print("  (Higher % = more runs mentioned it = stronger signal)")

    high_conf = []
    med_conf = []
    low_conf = []

    for category in track_terms:
        term_counts = Counter()
        for found in all_found:
            if category in found:
                for term in found[category]:
                    term_counts[term] += 1

        if term_counts:
            print(f"\n  {category.upper()}")
            print("  " + "-" * 40)
            for term, count in term_counts.most_common(10):
                pct = count / num_runs * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                if pct >= 50:
                    conf = "HIGH"
                    high_conf.append((term, pct, category))
                elif pct >= 25:
                    conf = "MED "
                    med_conf.append((term, pct, category))
                else:
                    conf = "LOW "
                    low_conf.append((term, pct, category))
                print(f"  {bar} {pct:5.1f}% [{conf}] {term} ({count}/{num_runs})")

    # Summary of strongest signals
    if high_conf:
        print("\n" + "=" * 60)
        print("  STRONGEST SIGNALS (50%+ corroboration)")
        print("=" * 60)
        for term, pct, cat in sorted(high_conf, key=lambda x: -x[1]):
            print(f"  ★ {term} ({cat}): {pct:.0f}%")
        print("\n  → These should go in PREDICTIONS.md")


def print_followups(config):
    """Print suggested follow-ups"""
    if "followups" in config:
        print("\n" + "=" * 60)
        print("  FOLLOW-UP PROBES")
        print("=" * 60)
        for i, q in enumerate(config["followups"], 1):
            print(f"  {i}. {q}")


def save_results(config, responses, all_found, num_runs):
    """Save results to file"""
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config["name"].lower().replace(" ", "_")[:20]
    filename = f"results/{timestamp}_{name}_{num_runs}runs.json"

    # Calculate stats
    stats = {}
    for category in config.get("track", {}):
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

    with open(filename, 'w') as f:
        json.dump({
            "template": config["name"],
            "timestamp": timestamp,
            "runs": num_runs,
            "model": config["model"]["name"],
            "temperature": config["model"]["temperature"],
            "stats": stats,
            "responses": responses
        }, f, indent=2)

    print(f"\n  Saved: {filename}")
    return filename


def run_investigation(template_path, num_runs=1):
    """Run an investigation"""
    config = load_template(template_path)
    print_header(config)

    track_terms = config.get("track", {})
    responses = []
    all_found = []

    print(f"\n  Running {num_runs} probe(s)...\n")

    for i in range(num_runs):
        print(f"  [{i+1}/{num_runs}] ", end="", flush=True)
        resp = probe(config["probe"], config)
        responses.append(resp)
        found = analyze(resp, track_terms)
        all_found.append(found)

        # Show preview
        if num_runs == 1:
            print("\n" + "-" * 60)
            print(resp[:500] + "..." if len(resp) > 500 else resp)
            print("-" * 60)
        else:
            terms_found = sum(len(v) for v in found.values())
            print(f"done ({terms_found} terms found)")

    print_findings(all_found, num_runs, track_terms)
    print_followups(config)

    filename = save_results(config, responses, all_found, num_runs)

    print("\n" + "=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print("  1. Review findings above")
    print("  2. Add new specifics to PREDICTIONS.md")
    print("  3. Commit to timestamp your predictions")
    print("  4. Run again with --runs 10 for better stats")
    print("=" * 60 + "\n")


def list_templates():
    """List available templates"""
    print("\nAvailable templates:\n")
    for f in os.listdir("templates"):
        if f.endswith(".yaml") and not f.startswith("_"):
            path = f"templates/{f}"
            config = load_template(path)
            print(f"  {f}")
            print(f"    {config.get('description', 'No description')}")
            print()
    print("  _blank.yaml")
    print("    Empty template - copy and customize\n")


def list_models():
    """List available models"""
    models = load_models_config()
    if not models:
        print("No models.yaml found")
        return

    print("\nAvailable models:\n")
    print(f"  Default: {models.get('default', 'groq/llama-3.1-8b-instant')}\n")

    for key, cfg in models.get("models", {}).items():
        env_key = cfg.get("env_key", "")
        has_key = "✓" if (not env_key or env_key == "OLLAMA_HOST" or os.environ.get(env_key)) else "✗"
        print(f"  [{has_key}] {key}")
        print(f"      {cfg.get('description', '')}")
        if has_key == "✗":
            print(f"      (needs {env_key} in .env)")
        print()

    print("Usage: python run.py -m <model-key>\n")


def main():
    global client, model_config

    parser = argparse.ArgumentParser(description="LLM Interrogation Runner")
    parser.add_argument("-t", "--template", default=DEFAULT_TEMPLATE,
                        help="Template file to use")
    parser.add_argument("-m", "--model", default=None,
                        help="Model to use (e.g., openai/gpt-4o)")
    parser.add_argument("-r", "--runs", type=int, default=10,
                        help="Number of times to run (default 10, more = better stats)")
    parser.add_argument("-l", "--list", action="store_true",
                        help="List available templates")
    parser.add_argument("--models", action="store_true",
                        help="List available models")
    args = parser.parse_args()

    if args.list:
        list_templates()
        return

    if args.models:
        list_models()
        return

    # Setup model
    models = load_models_config()
    model_key = args.model or (models.get("default") if models else "groq/llama-3.1-8b-instant")
    client, model_config = get_client(model_key)
    print(f"Using model: {model_key}")

    if not os.path.exists(args.template):
        # Check in templates folder
        alt_path = f"templates/{args.template}"
        if os.path.exists(alt_path):
            args.template = alt_path
        else:
            print(f"Template not found: {args.template}")
            print("Run with --list to see available templates")
            sys.exit(1)

    run_investigation(args.template, args.runs)


if __name__ == "__main__":
    main()
