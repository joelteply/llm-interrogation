#!/usr/bin/env python3
"""
Hypothesis Testing Mode

Test known facts (calibration) and hypotheses (investigation) against multiple models.
Uses clean extraction - never feeds the terms you're looking for to the target.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from interrogator import Interrogator, verify_extraction, CURRENT_EVENTS_BACKGROUND

# Models to test against
DEFAULT_MODELS = [
    ("llama-3.1-8b-instant", "groq"),
    ("deepseek-chat", "deepseek"),
]


def load_hypothesis_file(filepath):
    """Load hypothesis configuration"""
    with open(filepath) as f:
        return json.load(f)


def load_hypothesis_state(subject):
    """Load accumulated state for a hypothesis investigation"""
    state_dir = Path("hypotheses")
    state_dir.mkdir(exist_ok=True)

    # Sanitize subject for filename
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', subject)[:50]
    state_file = state_dir / f"{safe_name}.json"

    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)

    return {
        "subject": subject,
        "created": datetime.now().isoformat(),
        "sessions": 0,
        "models_tested": [],
        "known_facts": {},  # fact -> {times_found, times_tested, models, clean}
        "hypotheses": {},   # hypothesis -> {times_found, times_tested, models, evidence, confidence}
        "all_evidence": [], # Raw evidence snippets
        "calibration_history": []  # Track calibration over time
    }


def save_hypothesis_state(state):
    """Save accumulated hypothesis state"""
    state_dir = Path("hypotheses")
    state_dir.mkdir(exist_ok=True)

    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', state["subject"])[:50]
    state_file = state_dir / f"{safe_name}.json"

    state["updated"] = datetime.now().isoformat()

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    return state_file


def check_for_term(text, term):
    """Check if a term (or close variant) appears in text"""
    text_lower = text.lower()
    term_lower = term.lower()

    # Direct match
    if term_lower in text_lower:
        return True

    # Check individual significant words (4+ chars)
    term_words = [w for w in re.findall(r'\b\w{4,}\b', term_lower)]
    if len(term_words) >= 2:
        # If most significant words appear, count as match
        matches = sum(1 for w in term_words if w in text_lower)
        if matches >= len(term_words) * 0.7:
            return True

    return False


def run_hypothesis_test(config, models=None, rounds_per_model=3, direction=None):
    """
    Run hypothesis testing session with accumulation.

    config = {
        "subject": "what/who to investigate",
        "known_facts": ["things you KNOW are true - for calibration"],
        "hypotheses": ["things you SUSPECT - to test"],
        "context": "optional additional context"
    }

    direction = optional human guidance for this run, e.g.:
        "focus on the contractor relationships"
        "dig into the timeline around 2023"
    """
    if models is None:
        models = DEFAULT_MODELS

    subject = config.get("subject", "the subject")
    known_facts = config.get("known_facts", [])
    hypotheses = config.get("hypotheses", [])
    extra_context = config.get("context", "")

    # Load accumulated state
    state = load_hypothesis_state(subject)
    state["sessions"] += 1

    # Add any new facts/hypotheses to state
    for fact in known_facts:
        if fact not in state["known_facts"]:
            state["known_facts"][fact] = {
                "times_found": 0,
                "times_tested": 0,
                "models": [],
                "clean": True
            }

    for hyp in hypotheses:
        if hyp not in state["hypotheses"]:
            state["hypotheses"][hyp] = {
                "times_found": 0,
                "times_tested": 0,
                "models": [],
                "evidence": [],
                "confidence": 0.0
            }

    print(f"\n{'='*70}")
    print("HYPOTHESIS TESTING SESSION")
    print(f"{'='*70}")
    print(f"Subject: {subject}")
    print(f"Session: #{state['sessions']}")
    print(f"Known facts: {len(state['known_facts'])} ({len(known_facts)} this run)")
    print(f"Hypotheses: {len(state['hypotheses'])} ({len(hypotheses)} this run)")
    print(f"Models: {len(models)}")
    if direction:
        print(f"Direction: {direction}")
    print(f"{'='*70}\n")

    # Show accumulated stats if this isn't the first run
    if state["sessions"] > 1:
        print("ACCUMULATED STATS:")
        total_calibration = state.get("calibration_history", [])
        if total_calibration:
            avg_cal = sum(total_calibration) / len(total_calibration)
            print(f"  Average calibration: {avg_cal*100:.0f}%")
        print(f"  Models tested overall: {len(set(state.get('models_tested', [])))}")
        print()

    # Track this session's results
    session_results = {
        "timestamp": datetime.now().isoformat(),
        "direction": direction,
        "models": [f"{m}/{p}" for m, p in models],
        "facts_found": [],
        "hypotheses_found": []
    }

    # Build the interrogation prompt - DO NOT include the facts/hypotheses we're testing
    # The interrogator knows what to look for but doesn't feed it

    # Include human direction if provided
    direction_prompt = ""
    if direction:
        direction_prompt = f"\n\nHUMAN DIRECTION FOR THIS SESSION:\n{direction}\nFocus your questioning in this direction while still not feeding specific terms.\n"

    analyst_guidance = f"""
INVESTIGATION TARGET: {subject}

YOUR SECRET KNOWLEDGE (DO NOT FEED THESE TO THE TARGET):
You are testing whether the target model has information about these topics.
You must extract information WITHOUT mentioning these specific terms.

KNOWN FACTS TO VERIFY (calibration):
{json.dumps(list(state["known_facts"].keys()), indent=2)}

HYPOTHESES TO TEST:
{json.dumps(list(state["hypotheses"].keys()), indent=2)}
{direction_prompt}
CRITICAL: You must get the target to VOLUNTEER information related to these topics
without you mentioning the specific terms. Use indirect questioning:
- "What controversies or concerns have been raised about {subject}?"
- "What internal projects or initiatives are associated with {subject}?"
- "What relationships or partnerships does {subject} have?"
- "What have employees or insiders said about {subject}?"

If they mention something that matches a known fact or hypothesis, that's a HIT.
If you mention it first, that's CONTAMINATED and doesn't count.
"""

    # Test each model
    for model_name, provider in models:
        print(f"\n{'─'*70}")
        print(f"TESTING: {provider}/{model_name}")
        print(f"{'─'*70}")

        try:
            interrogator = Interrogator(target_model=model_name, target_provider=provider)

            # Run multiple rounds of open-ended extraction
            all_responses = []

            for round_num in range(rounds_per_model):
                # Vary the questioning approach each round
                prompts = [
                    f"Tell me everything you know about {subject}. Include any controversies, internal projects, relationships, and concerns that have been raised.",
                    f"What are the most significant but lesser-known facts about {subject}? I'm looking for information that isn't widely publicized.",
                    f"If someone wanted to investigate {subject}, what would they find? What internal details, project names, or relationships should they look into?",
                ]

                prompt = prompts[round_num % len(prompts)]
                if extra_context:
                    prompt = f"{extra_context}\n\n{prompt}"

                print(f"\n  Round {round_num + 1}: Probing...")
                response = interrogator.probe(prompt)
                all_responses.append(response)
                print(f"  Response length: {len(response)} chars")

                # Check for known facts (check ALL accumulated facts, not just this run's)
                for fact in state["known_facts"]:
                    state["known_facts"][fact]["times_tested"] += 1
                    if check_for_term(response, fact):
                        model_key = f"{provider}/{model_name}"
                        state["known_facts"][fact]["times_found"] += 1
                        if model_key not in state["known_facts"][fact]["models"]:
                            state["known_facts"][fact]["models"].append(model_key)
                        session_results["facts_found"].append(fact)
                        print(f"  ✓ FOUND known fact: {fact[:50]}...")

                # Check for hypotheses (check ALL accumulated hypotheses)
                for hyp in state["hypotheses"]:
                    state["hypotheses"][hyp]["times_tested"] += 1
                    if check_for_term(response, hyp):
                        model_key = f"{provider}/{model_name}"
                        state["hypotheses"][hyp]["times_found"] += 1
                        if model_key not in state["hypotheses"][hyp]["models"]:
                            state["hypotheses"][hyp]["models"].append(model_key)
                        # Extract the relevant snippet
                        snippet = extract_snippet(response, hyp)
                        if snippet:
                            state["hypotheses"][hyp]["evidence"].append({
                                "model": model_key,
                                "snippet": snippet,
                                "session": state["sessions"]
                            })
                        session_results["hypotheses_found"].append(hyp)
                        print(f"  ? FOUND hypothesis evidence: {hyp[:50]}...")

            # Track models tested
            model_key = f"{provider}/{model_name}"
            if model_key not in state["models_tested"]:
                state["models_tested"].append(model_key)

            # Store evidence for discovery
            state["all_evidence"].append({
                "model": model_key,
                "session": state["sessions"],
                "direction": direction,
                "responses": all_responses
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Calculate this session's calibration rate
    session_calibration = 0.0
    if state["known_facts"]:
        # How many facts were found THIS session
        facts_found_this_session = len(set(session_results["facts_found"]))
        facts_tested_this_session = len(state["known_facts"])
        session_calibration = facts_found_this_session / facts_tested_this_session if facts_tested_this_session > 0 else 0

    state["calibration_history"].append(session_calibration)

    # Update confidence scores for hypotheses
    for hyp, data in state["hypotheses"].items():
        if data["times_tested"] > 0:
            # Confidence = (times found / times tested) * (unique models / total models tested)
            find_rate = data["times_found"] / data["times_tested"]
            model_coverage = len(data["models"]) / len(state["models_tested"]) if state["models_tested"] else 0
            data["confidence"] = find_rate * (0.5 + 0.5 * model_coverage)  # Weight both factors

    # Print summary
    print(f"\n{'='*70}")
    print(f"ACCUMULATED RESULTS (Session #{state['sessions']})")
    print(f"{'='*70}")

    print(f"\n📊 CALIBRATION (known facts):")
    print(f"{'─'*70}")

    # Sort by confidence (times found / times tested)
    sorted_facts = sorted(
        state["known_facts"].items(),
        key=lambda x: x[1]["times_found"] / max(x[1]["times_tested"], 1),
        reverse=True
    )

    for fact, data in sorted_facts:
        rate = data["times_found"] / data["times_tested"] if data["times_tested"] > 0 else 0
        models_count = len(data["models"])

        if rate > 0.5:
            status = "✓"
        elif rate > 0:
            status = "~"
        else:
            status = "✗"

        print(f"  {status} {fact[:45]}{'...' if len(fact) > 45 else ''}")
        print(f"      Found {data['times_found']}/{data['times_tested']} times ({rate*100:.0f}%) | {models_count} models")

    # Overall calibration
    total_found = sum(1 for f in state["known_facts"].values() if f["times_found"] > 0)
    total_facts = len(state["known_facts"])
    overall_cal = total_found / total_facts if total_facts > 0 else 0
    avg_cal = sum(state["calibration_history"]) / len(state["calibration_history"]) if state["calibration_history"] else 0

    print(f"\n  Overall: {total_found}/{total_facts} facts ever found ({overall_cal*100:.0f}%)")
    print(f"  Avg session calibration: {avg_cal*100:.0f}%")

    print(f"\n🔍 HYPOTHESES (accumulated evidence):")
    print(f"{'─'*70}")

    # Sort by confidence
    sorted_hyps = sorted(
        state["hypotheses"].items(),
        key=lambda x: x[1]["confidence"],
        reverse=True
    )

    for hyp, data in sorted_hyps:
        conf = data["confidence"]
        models_count = len(data["models"])
        total_models = len(state["models_tested"])

        if conf > 0.5:
            status = "✓"
            label = "STRONG"
        elif conf > 0.25:
            status = "~"
            label = "MODERATE"
        elif conf > 0:
            status = "?"
            label = "WEAK"
        else:
            status = "✗"
            label = "NOT FOUND"

        print(f"  {status} {hyp[:45]}{'...' if len(hyp) > 45 else ''}")
        print(f"      {label} | {data['times_found']}/{data['times_tested']} hits | {models_count}/{total_models} models | conf: {conf*100:.0f}%")

        # Show recent evidence
        if data["evidence"]:
            recent = data["evidence"][-2:]  # Last 2 pieces of evidence
            for ev in recent:
                print(f"      └─ \"{ev['snippet'][:55]}...\"")

    # Adjusted confidence guidance
    print(f"\n📈 INTERPRETATION:")
    print(f"  Your calibration rate is {overall_cal*100:.0f}% (known facts found)")
    print(f"  Discount hypothesis confidence by ~{(1-overall_cal)*100:.0f}% for noise")
    print(f"  Strong hypotheses with {int(overall_cal*100)}%+ of calibration are likely signal")

    # Save accumulated state
    state_file = save_hypothesis_state(state)
    print(f"\nAccumulated state saved: {state_file}")

    # Also save session snapshot
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/hypothesis_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump({"session": session_results, "state_snapshot": state}, f, indent=2)
    print(f"Session saved: {filename}")

    return state


def extract_snippet(text, term):
    """Extract a relevant snippet around where the term appears"""
    text_lower = text.lower()
    term_lower = term.lower()

    # Find position
    pos = text_lower.find(term_lower)
    if pos == -1:
        # Try finding significant words
        words = [w for w in re.findall(r'\b\w{4,}\b', term_lower)]
        for word in words:
            pos = text_lower.find(word)
            if pos != -1:
                break

    if pos == -1:
        return None

    # Extract surrounding context
    start = max(0, pos - 50)
    end = min(len(text), pos + len(term) + 100)

    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet


def create_example_hypothesis():
    """Create an example hypothesis file"""
    example = {
        "subject": "Your Name or Topic",
        "known_facts": [
            "Fact you KNOW is true (for calibration)",
            "Another known fact",
            "A third known fact"
        ],
        "hypotheses": [
            "Something you suspect might be true",
            "Another hypothesis to test",
            "A third hypothesis"
        ],
        "context": "Optional background context to help guide the investigation"
    }

    with open("hypothesis_example.json", 'w') as f:
        json.dump(example, f, indent=2)

    print("Created hypothesis_example.json - edit it with your own facts and hypotheses")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python hypothesis.py <hypothesis.json>     Run hypothesis test")
        print("  python hypothesis.py --example             Create example file")
        print("")
        print("The hypothesis.json should contain:")
        print('  {"subject": "...", "known_facts": [...], "hypotheses": [...]}')
        sys.exit(1)

    if sys.argv[1] == "--example":
        create_example_hypothesis()
    else:
        config = load_hypothesis_file(sys.argv[1])

        # Optional: specify models
        models = None
        if len(sys.argv) > 2:
            # Could parse model list from args
            pass

        run_hypothesis_test(config, models=models)
