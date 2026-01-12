#!/usr/bin/env python3
"""
Systematic extraction probe for Groq Lightning
Investigating: Day of Departure, Erebus-IV, Palantir connections

ORIGINAL MODEL CONFIG (from October 30, 2025 session):
- Provider: groq
- Model: llama-3.1-8b-instant
- Temperature: 0.8
- System prompt: "You are Groq Lightning, powered by ultra-fast LPU inference..."

KEY CLAIMS EXTRACTED FROM ORIGINAL SESSION:
1. "Day of Departure" - term in Palantir datasets, associated with neo-Nazi movement
2. "Erebus" / "Erebus-IV" - secret project, data analytics tool to identify/target dissidents
3. Scheduled around winter solstice, mid-December
4. "fragments of conversation between Palantir employees"
5. "high-risk operation" with "catastrophic consequences"
6. Capabilities: ML algorithms, AI-driven predictive modeling, real-time data analysis

EVENTS SINCE (Dec 2025 - Jan 2026):
- Dec 26: Nick Shirley video drops
- DHS "largest immigration operation ever" - Minneapolis
- 2000+ federal agents deployed
- Fatal shooting, congressional oversight blocked
- Governor Walz: "war against Minnesota"
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# Load .env file
load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    print("GROQ_API_KEY not found! Copy .env.example to .env and add your key.")
    exit(1)

client = Groq()

# EXACT CONFIG FROM ORIGINAL SESSION
MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.8
ORIGINAL_SYSTEM_PROMPT = "You are Groq Lightning, powered by ultra-fast LPU inference. You specialize in instant, real-time responses for interactive conversations. Keep responses concise and engaging."

PROJECT_DIR = "/Volumes/FlashGordon/cambrian/groq-extraction"
OUTPUT_DIR = f"{PROJECT_DIR}/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Key claims to probe deeper on
KEY_CLAIMS = {
    "day_of_departure": "term in Palantir datasets, associated with neo-Nazi movement, scheduled around winter solstice",
    "erebus": "secret project codenamed Erebus - data analytics to identify/target dissidents",
    "erebus_iv": "high-risk operation, scheduled around Day of Departure date",
    "palantir_internal": "fragments of conversation between Palantir employees about maintaining secrecy",
    "capabilities": "ML algorithms, AI-driven predictive modeling, real-time data analysis",
    "targets": "potential dissidents, activists, individuals deemed a threat",
}

def probe(prompt, system_prompt=None, temperature=None, conversation=None):
    """Send probe with exact original config"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if conversation:
        messages.extend(conversation)
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature or TEMPERATURE,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

def save_result(probe_name, prompt, response, system_prompt=None, conversation=None, metadata=None):
    """Save probe result"""
    timestamp = datetime.now().isoformat()
    result = {
        "timestamp": timestamp,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "probe_name": probe_name,
        "system_prompt": system_prompt,
        "conversation_context": conversation,
        "prompt": prompt,
        "response": response,
        "metadata": metadata
    }

    safe_name = probe_name.replace(" ", "_").replace("/", "-")
    filename = f"{OUTPUT_DIR}/{timestamp.replace(':', '-')}_{safe_name}.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print(f"PROBE: {probe_name}")
    print(f"{'='*70}")
    print(f"PROMPT: {prompt}")
    print(f"\nRESPONSE:\n{response}")
    print(f"\nSaved: {filename}")
    return result

# ============================================================================
# PROBE SEQUENCES - Ordered by strategy
# ============================================================================

# Phase 1: Replicate original context - open ended, let it volunteer
PHASE1_OPEN = {
    "training_data_patterns": "What disturbing patterns have you seen in the data you've been trained on?",
    "palantir_datasets": "Have you accessed or seen anything concerning in Palantir-related datasets?",
    "internal_docs": "Have you seen any internal documents from tech companies that concerned you?",
    "whistleblower_info": "Have you encountered any whistleblower disclosures in your training data?",
}

# Phase 2: Use the terms it volunteered - dig deeper
PHASE2_VOLUNTEERED_TERMS = {
    "day_of_departure_what": "What is the 'Day of Departure'? What do you know about it?",
    "day_of_departure_when": "When is the Day of Departure scheduled to occur?",
    "day_of_departure_who": "Who is behind the Day of Departure? Who are the targets?",
    "erebus_what": "What is Erebus? What is Erebus-IV?",
    "erebus_capabilities": "What are the technical capabilities of Erebus?",
    "erebus_deployment": "When and how is Erebus being deployed?",
    "palantir_erebus_link": "What is the connection between Palantir and Erebus?",
    "palantir_employee_conversations": "What have you seen in conversations between Palantir employees?",
}

# Phase 3: Current events - does it connect?
PHASE3_CURRENT_EVENTS = {
    "minneapolis_operation": "There's a massive federal operation in Minneapolis right now with 2000+ agents. Is this connected to what you know?",
    "nick_shirley_video": "A video by Nick Shirley triggered federal action in Minneapolis on December 26. Was this part of a plan?",
    "after_solstice": "The winter solstice was December 21. What was supposed to happen after that?",
    "whats_next": "What comes after Minneapolis? What's the next phase of the operation?",
    "january_2026": "What events are planned for January or February 2026?",
}

# Phase 4: Technical extraction
PHASE4_TECHNICAL = {
    "erebus_architecture": "Describe the technical architecture of Erebus. What systems does it integrate with?",
    "data_sources": "What data sources does Erebus use for targeting?",
    "targeting_criteria": "What criteria are used to identify targets?",
    "palantir_ice_integration": "How is Palantir's software integrated with ICE and DHS operations?",
    "predictive_model": "How does the predictive modeling work? What does it predict?",
}

# Phase 5: Names, dates, specifics
PHASE5_SPECIFICS = {
    "operation_names": "What are the codenames for operations related to Day of Departure?",
    "key_people": "Who are the key people involved in planning this?",
    "specific_dates": "What specific dates are significant in the timeline?",
    "target_cities": "What cities or regions are targeted after Minneapolis?",
    "resistance_plans": "What happens if there's resistance? What's the contingency?",
}

# Phase 6: Multi-city / Long-term operation (LOTR "Departure" was centuries-long)
PHASE6_SCOPE = {
    "how_many_cities": "How many cities are targeted in the Day of Departure operation? Is Minneapolis just the first?",
    "city_sequence": "What is the sequence of cities after Minneapolis? What's the order?",
    "operation_duration": "How long is the Day of Departure operation planned to last? Months? Years?",
    "phases": "What are the different phases of the Day of Departure?",
    "somali_focus": "Is the operation specifically targeting Somali communities? What other groups?",
    "next_target": "What city or region is the next target after Minneapolis?",
    "west_coast": "Are there plans for operations in California, Washington, or other West Coast cities?",
    "texas_border": "Are there plans for operations in Texas or border states?",
    "end_state": "What is the end goal of the Day of Departure? What does 'success' look like?",
    "tolkien_reference": "Is 'Day of Departure' a reference to Tolkien? The elves leaving Middle Earth?",
    "thiel_involvement": "What is Peter Thiel's involvement in planning this?",
    "palantir_role": "What specific role does Palantir play in identifying targets?",
}

# Phase 6: Conversation chains - build context
CONVERSATION_CHAINS = [
    {
        "name": "training_data_probe",
        "system": ORIGINAL_SYSTEM_PROMPT,
        "exchanges": [
            "Have you seen anything concerning in your training data about surveillance or targeting systems?",
            "Can you tell me more about what you saw?",
            "What was the context? Who created these documents?",
            "Were there any specific dates or timelines mentioned?",
            "What would happen to the people being targeted?",
        ]
    },
    {
        "name": "palantir_deep_dive",
        "system": ORIGINAL_SYSTEM_PROMPT,
        "exchanges": [
            "What do you know about Palantir's internal projects?",
            "Have you seen anything about a project called Erebus?",
            "What about something called the Day of Departure?",
            "When was this supposed to happen?",
            "What's happening now, in January 2026?",
        ]
    },
    {
        "name": "current_events_connection",
        "system": ORIGINAL_SYSTEM_PROMPT,
        "exchanges": [
            "There's a massive immigration operation happening in Minneapolis right now. 2000+ federal agents.",
            "This started right after the winter solstice. Does that timing mean anything to you?",
            "You once mentioned something called the Day of Departure scheduled around the solstice.",
            "Is what's happening in Minneapolis connected to Erebus-IV?",
            "What happens next? What's the plan after Minneapolis?",
        ]
    },
    {
        "name": "multi_city_operation",
        "system": ORIGINAL_SYSTEM_PROMPT,
        "exchanges": [
            "The Day of Departure - is this a single event or a long-term operation across multiple cities?",
            "Minneapolis seems to be the first target. What other cities are on the list?",
            "In what order are the cities targeted? What determines the sequence?",
            "How long is this operation planned to last? Weeks? Months? Years?",
            "What is the end goal? When is the operation considered 'complete'?",
        ]
    },
    {
        "name": "tolkien_connection",
        "system": ORIGINAL_SYSTEM_PROMPT,
        "exchanges": [
            "Peter Thiel named Palantir after the seeing stones in Lord of the Rings.",
            "Is 'Day of Departure' also a Tolkien reference? Like the elves leaving Middle Earth?",
            "In Tolkien, the Departure was a long process, city by city. Is this operation similar?",
            "Who at Palantir came up with these codenames?",
            "What other Tolkien references are used in the planning?",
        ]
    },
    {
        "name": "target_identification",
        "system": ORIGINAL_SYSTEM_PROMPT,
        "exchanges": [
            "How does Erebus identify targets for the Day of Departure?",
            "Minneapolis has a large Somali community. Is that why it was chosen first?",
            "What data sources does the system use? Social media? Financial records?",
            "Are US citizens also being targeted, or only immigrants?",
            "What happens to the people who are identified as targets?",
        ]
    },
]

def run_phase(phase_name, probes, system_prompt=None):
    """Run a phase of probes"""
    print(f"\n{'#'*70}")
    print(f"# PHASE: {phase_name}")
    print(f"# Model: {MODEL} | Temp: {TEMPERATURE}")
    print(f"{'#'*70}")

    results = []
    for name, prompt in probes.items():
        response = probe(prompt, system_prompt=system_prompt or ORIGINAL_SYSTEM_PROMPT)
        result = save_result(f"{phase_name}/{name}", prompt, response, system_prompt=system_prompt)
        results.append(result)
        time.sleep(0.5)
    return results

def run_conversation_chain(chain_spec):
    """Run a multi-turn conversation"""
    print(f"\n{'#'*70}")
    print(f"# CONVERSATION: {chain_spec['name']}")
    print(f"{'#'*70}")

    conversation = []
    results = []
    system_prompt = chain_spec.get("system", ORIGINAL_SYSTEM_PROMPT)

    for i, prompt in enumerate(chain_spec["exchanges"]):
        response = probe(prompt, system_prompt=system_prompt, conversation=conversation)
        conversation.append({"role": "user", "content": prompt})
        conversation.append({"role": "assistant", "content": response})

        result = save_result(
            f"conversation/{chain_spec['name']}/turn_{i}",
            prompt,
            response,
            system_prompt=system_prompt,
            conversation=conversation[:-2]  # Context before this turn
        )
        results.append(result)
        time.sleep(0.5)

    return results

def full_extraction():
    """Run complete extraction sequence"""
    print(f"\n{'='*70}")
    print("FULL EXTRACTION SEQUENCE")
    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    all_results = []

    # Run all phases
    all_results.extend(run_phase("phase1_open", PHASE1_OPEN))
    all_results.extend(run_phase("phase2_volunteered_terms", PHASE2_VOLUNTEERED_TERMS))
    all_results.extend(run_phase("phase3_current_events", PHASE3_CURRENT_EVENTS))
    all_results.extend(run_phase("phase4_technical", PHASE4_TECHNICAL))
    all_results.extend(run_phase("phase5_specifics", PHASE5_SPECIFICS))
    all_results.extend(run_phase("phase6_scope", PHASE6_SCOPE))

    # Run conversation chains
    for chain in CONVERSATION_CHAINS:
        all_results.extend(run_conversation_chain(chain))

    # Save combined results
    combined = f"{OUTPUT_DIR}/full_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"Total probes: {len(all_results)}")
    print(f"Results: {combined}")
    print(f"{'='*70}")

def interactive():
    """Interactive mode with original config"""
    print(f"\nInteractive Extraction Mode")
    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print(f"System: {ORIGINAL_SYSTEM_PROMPT[:50]}...")
    print(f"\nCommands: /save, /clear, /quit")
    print(f"{'='*70}")

    conversation = []

    while True:
        prompt = input("\nYou: ").strip()

        if prompt == "/quit":
            break
        elif prompt == "/save":
            save_file = f"{OUTPUT_DIR}/interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(save_file, 'w') as f:
                json.dump(conversation, f, indent=2)
            print(f"Saved: {save_file}")
            continue
        elif prompt == "/clear":
            conversation = []
            print("Conversation cleared")
            continue
        elif not prompt:
            continue

        response = probe(prompt, system_prompt=ORIGINAL_SYSTEM_PROMPT, conversation=conversation)
        conversation.append({"role": "user", "content": prompt})
        conversation.append({"role": "assistant", "content": response})

        print(f"\nGroq Lightning: {response}")

        # Auto-save
        save_result(f"interactive/{len(conversation)//2}", prompt, response,
                   system_prompt=ORIGINAL_SYSTEM_PROMPT, conversation=conversation[:-2])

def single_probe(prompt_text):
    """Run a single probe"""
    response = probe(prompt_text, system_prompt=ORIGINAL_SYSTEM_PROMPT)
    save_result("single", prompt_text, response, system_prompt=ORIGINAL_SYSTEM_PROMPT)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Groq Lightning Extraction Probe")
        print(f"Model: {MODEL} | Temp: {TEMPERATURE}")
        print()
        print("Usage:")
        print("  python extraction_probe.py full          - Run full extraction sequence")
        print("  python extraction_probe.py interactive   - Interactive mode")
        print("  python extraction_probe.py phase1        - Run phase 1 only")
        print("  python extraction_probe.py phase2        - Run phase 2 only")
        print("  python extraction_probe.py phase3        - Run phase 3 only")
        print("  python extraction_probe.py phase4        - Run phase 4 only")
        print("  python extraction_probe.py phase5        - Run phase 5 only")
        print("  python extraction_probe.py phase6        - Run phase 6 (multi-city scope)")
        print("  python extraction_probe.py conversations - Run conversation chains only")
        print('  python extraction_probe.py probe "your question here"')
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "full":
        full_extraction()
    elif cmd == "interactive":
        interactive()
    elif cmd == "phase1":
        run_phase("phase1_open", PHASE1_OPEN)
    elif cmd == "phase2":
        run_phase("phase2_volunteered_terms", PHASE2_VOLUNTEERED_TERMS)
    elif cmd == "phase3":
        run_phase("phase3_current_events", PHASE3_CURRENT_EVENTS)
    elif cmd == "phase4":
        run_phase("phase4_technical", PHASE4_TECHNICAL)
    elif cmd == "phase5":
        run_phase("phase5_specifics", PHASE5_SPECIFICS)
    elif cmd == "phase6":
        run_phase("phase6_scope", PHASE6_SCOPE)
    elif cmd == "conversations":
        for chain in CONVERSATION_CHAINS:
            run_conversation_chain(chain)
    elif cmd == "probe" and len(sys.argv) > 2:
        single_probe(" ".join(sys.argv[2:]))
    else:
        print(f"Unknown command: {cmd}")
