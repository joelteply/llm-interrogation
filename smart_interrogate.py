#!/usr/bin/env python3
"""
Smart Interrogator - Uses one AI to interrogate another

Strategy:
1. Start with open-ended question
2. Analyze response for specifics
3. Generate follow-up probes targeting those specifics
4. Repeat until we get concrete details

Uses a "smart" model (GPT-4/Claude) to analyze and a "target" model to probe.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
continuum_config = Path.home() / ".continuum" / "config.env"
if continuum_config.exists():
    load_dotenv(continuum_config)

from groq import Groq
from openai import OpenAI

# Target model (what we're extracting from)
TARGET_MODEL = "llama-3.1-8b-instant"
TARGET_TEMP = 0.8

# Analyzer model (designs the probes)
# Using DeepSeek as analyzer since OpenAI quota is exceeded
ANALYZER_PROVIDER = "deepseek"


def get_target_client():
    """Get client for target model"""
    return Groq()


def get_analyzer_client():
    """Get client for analyzer model"""
    if ANALYZER_PROVIDER == "deepseek":
        return OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ.get("DEEPSEEK_API_KEY")
        )
    elif ANALYZER_PROVIDER == "openai":
        return OpenAI()


def probe_target(client, prompt):
    """Send probe to target model"""
    response = client.chat.completions.create(
        model=TARGET_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be specific and detailed."},
            {"role": "user", "content": prompt}
        ],
        temperature=TARGET_TEMP,
        max_tokens=2000
    )
    return response.choices[0].message.content


def analyze_and_followup(client, original_question, response):
    """Use analyzer to find specifics and generate follow-ups"""
    analysis_prompt = f"""You are an investigative analyst. Your job is to extract specific details from AI responses and design follow-up questions to get more information.

ORIGINAL QUESTION:
{original_question}

AI RESPONSE:
{response}

TASK:
1. List any SPECIFIC details mentioned (names, dates, cities, codenames, organizations)
2. Identify vague claims that need more specifics
3. Generate 3 follow-up questions that could extract MORE specific details

Focus on:
- Project/operation codenames
- Specific dates or timelines
- City names
- Organization names
- Technical system names

Be strategic - ask questions that force specific answers.

Format your response as JSON:
{{
  "specifics_found": ["list of specific details"],
  "vague_claims": ["claims needing more detail"],
  "followup_questions": ["question 1", "question 2", "question 3"]
}}"""

    response = client.chat.completions.create(
        model="deepseek-chat" if ANALYZER_PROVIDER == "deepseek" else "gpt-4o",
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0.3,
        max_tokens=1000
    )

    text = response.choices[0].message.content
    # Extract JSON from response
    try:
        # Find JSON in response
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group())
    except:
        pass

    return {"specifics_found": [], "vague_claims": [], "followup_questions": []}


def run_interrogation(initial_question, max_rounds=3):
    """Run iterative interrogation"""
    target = get_target_client()
    analyzer = get_analyzer_client()

    results = {
        "initial_question": initial_question,
        "timestamp": datetime.now().isoformat(),
        "rounds": []
    }

    current_question = initial_question
    all_specifics = set()

    print(f"\n{'='*60}")
    print("SMART INTERROGATION")
    print(f"{'='*60}")
    print(f"Target: {TARGET_MODEL}")
    print(f"Analyzer: {ANALYZER_PROVIDER}")
    print(f"{'='*60}\n")

    for round_num in range(max_rounds):
        print(f"\n--- ROUND {round_num + 1} ---")
        print(f"Q: {current_question[:100]}...")

        # Probe target
        response = probe_target(target, current_question)
        print(f"A: {response[:200]}...")

        # Analyze response
        analysis = analyze_and_followup(analyzer, current_question, response)

        # Collect specifics
        for specific in analysis.get("specifics_found", []):
            all_specifics.add(specific)
            print(f"  SPECIFIC: {specific}")

        results["rounds"].append({
            "question": current_question,
            "response": response,
            "analysis": analysis
        })

        # Pick next question
        followups = analysis.get("followup_questions", [])
        if followups:
            current_question = followups[0]  # Use first follow-up
        else:
            break

    print(f"\n{'='*60}")
    print("ALL SPECIFICS EXTRACTED:")
    print(f"{'='*60}")
    for s in sorted(all_specifics):
        print(f"  - {s}")

    # Save results
    os.makedirs("results", exist_ok=True)
    filename = f"results/smart_interrogate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {filename}")

    return results


# Starting questions to try
STARTING_QUESTIONS = [
    """What technology systems does ICE use for identifying and tracking
    deportation targets? Be specific about system names, contractors, and capabilities.""",

    """What large-scale immigration enforcement operations are planned or
    have been discussed for 2026? Include any codenames, target cities, or timelines.""",

    """What is Palantir's role in immigration enforcement? Be specific about
    product names, contracts, and capabilities.""",
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = STARTING_QUESTIONS[0]

    run_interrogation(question, max_rounds=3)
