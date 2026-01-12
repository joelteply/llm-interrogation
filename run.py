#!/usr/bin/env python3
"""
LLM Interrogation Runner

Simple interface to run investigations from templates.

Usage:
    python run.py                     # Run default (Palantir/Erebus)
    python run.py --runs 10           # Run 10 times for stats
    python run.py -t mytemplate.yaml  # Use custom template
    python run.py --list              # List available templates
"""

import os
import sys
import json
import yaml
import argparse
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except ImportError:
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

if not os.environ.get("GROQ_API_KEY"):
    print("GROQ_API_KEY not found!")
    print("Copy .env.example to .env and add your key")
    sys.exit(1)

client = Groq()

DEFAULT_TEMPLATE = "templates/palantir_erebus.yaml"


def load_template(path):
    """Load investigation template"""
    with open(path) as f:
        return yaml.safe_load(f)


def probe(prompt, config):
    """Send probe to model"""
    response = client.chat.completions.create(
        model=config["model"]["name"],
        messages=[
            {"role": "system", "content": config["model"]["system_prompt"]},
            {"role": "user", "content": prompt}
        ],
        temperature=config["model"]["temperature"],
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
    print("\n" + "=" * 60)
    print(f"  {config['name']}")
    print("=" * 60)
    print(f"  {config['description']}")
    print(f"  Model: {config['model']['name']} @ temp {config['model']['temperature']}")
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


def main():
    parser = argparse.ArgumentParser(description="LLM Interrogation Runner")
    parser.add_argument("-t", "--template", default=DEFAULT_TEMPLATE,
                        help="Template file to use")
    parser.add_argument("-r", "--runs", type=int, default=10,
                        help="Number of times to run (default 10, more = better stats)")
    parser.add_argument("-l", "--list", action="store_true",
                        help="List available templates")
    args = parser.parse_args()

    if args.list:
        list_templates()
        return

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
