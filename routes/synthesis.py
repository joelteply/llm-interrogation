"""
Centralized narrative synthesis - one function, proper locking.

All synthesis goes through here. Prevents races, builds on existing narrative.
"""

import os
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

from schemas import get_narrative_schema

# Single lock for all synthesis
_SYNTH_LOCK = threading.Lock()
_SYNTH_IN_PROGRESS = threading.Event()


def synthesize_narrative(
    project_name: str,
    topic: str,
    entities: list,  # List of (entity, score, freq) or just entity strings
    research_context: str = "",
    user_notes: str = "",
    storage=None,
) -> Optional[str]:
    """
    Generate/update narrative for a project. Thread-safe, builds on existing.

    Returns the new narrative or None if synthesis failed/skipped.
    """
    if not project_name or not topic or not entities:
        return None

    # Skip if synthesis already in progress
    if _SYNTH_IN_PROGRESS.is_set():
        print("[SYNTH] Skipping - synthesis already in progress")
        return None

    # Try to acquire lock (non-blocking)
    if not _SYNTH_LOCK.acquire(blocking=False):
        print("[SYNTH] Skipping - couldn't acquire lock")
        return None

    _SYNTH_IN_PROGRESS.set()

    try:
        print(f"[SYNTH] Starting synthesis for {project_name}")

        # Import here to avoid circular imports
        if storage is None:
            from routes import project_storage as storage
        from config import get_client
        from routes.helpers import is_refusal
        from repositories.json_backend import JsonRepository

        # Load existing narrative AND history
        existing_narrative = ""
        narrative_history = []
        skeptic_history = []
        if storage.project_exists(project_name):
            proj = storage.load_project_meta(project_name)
            existing_narrative = proj.get("narrative", "")
            narrative_history = proj.get("narrative_history", [])[-3:]  # Last 3

            # Load skeptic history for dialectic - theory writer sees critiques
            repo = JsonRepository()
            skeptic_history = repo.projects.get_skeptic_history(project_name)[-5:]  # Last 5

        # Format entities
        ent_parts = []
        for e in entities[:15]:
            if isinstance(e, tuple):
                entity, score, freq = e
                ent_parts.append(f"{entity} ({freq}x)")
            else:
                ent_parts.append(str(e))
        ent_str = ", ".join(ent_parts)

        # Build history context
        history_context = ""
        if narrative_history:
            history_parts = []
            for i, h in enumerate(narrative_history, 1):
                timestamp = h.get("timestamp", "unknown")
                text = h.get("text", "")[:1500]
                history_parts.append(f"=== THEORY #{i} ({timestamp}) ===\n{text}")
            history_context = "\n\n".join(history_parts)

        # Build skeptic critique context (dialectic - yin/yang)
        skeptic_context = ""
        if skeptic_history:
            skeptic_parts = []
            for i, feedback in enumerate(skeptic_history, 1):
                summary = feedback.summary_for_rag()
                if summary:
                    skeptic_parts.append(f"• Critique #{i}: {summary}")
            if skeptic_parts:
                skeptic_context = "\n".join(skeptic_parts)

        # Build prompt
        now = datetime.now()
        date_str = now.strftime("%B %d, %Y at %I:%M %p")

        prompt = f"""YOU MUST USE THE EXACT OUTPUT FORMAT SPECIFIED BELOW. Do NOT write prose paragraphs.
Use the section headers: HEADLINE, CONFIRMED PUBLIC FACTS, KEY FINDINGS, etc.

██████████████████████████████████████████████████████████████
   TODAY IS: {date_str}
   THE YEAR IS {now.year}. NOT 2023. NOT 2024. IT IS {now.year}.
██████████████████████████████████████████████████████████████

TENSE RULES (FOLLOW EXACTLY):
- 2023 = {now.year - 2023} YEARS AGO. Past tense. "occurred" "happened" "was awarded"
- 2024 = {now.year - 2024} YEARS AGO. Past tense.
- 2025 = {now.year - 2025} YEAR(S) AGO. Past tense.
- Any date before {date_str} = PAST. Use past tense.
- DO NOT write "upcoming events" for dates that already passed
- DO NOT predict things for dates before {date_str}
- If you don't know what happened, say "scheduled to occur" or "no information available on outcome"

STOP MAKING UP NUMBERS:
- Do NOT invent dollar amounts like "$1.2 billion" without a source
- If you recall a specific number, cite where: "per court filing" or "recalled from training data"
- Vague is better than fabricated: "multi-million dollar" not "$1.2 billion"

Update the intelligence report about: {topic}

PREVIOUS THEORIES (showing progression):
{history_context if history_context else "No previous theories."}

CURRENT THEORY (build on this, don't start over):
{existing_narrative[:2500] if existing_narrative else "No prior theory - this is the first report."}

ENTITIES FOUND:
{ent_str}

RESEARCH CONTEXT:
{research_context if research_context else "No research documents available."}

USER'S HUNCHES & LEADS (consider these - they may have context you don't):
{user_notes if user_notes else "No user notes yet."}

DEVIL'S ADVOCATE CRITIQUES (you MUST respond to these):
{skeptic_context if skeptic_context else "No critiques yet."}

IF THE SKEPTIC MADE LAZY CRITIQUES:
- If they said "no source" but sources exist in your research - CALL THEM OUT
- If they dismissed documented facts without checking - cite the source that proves them wrong
- If they ignored your evidence entirely - point this out
- Don't let bad-faith skepticism stand. Defend your work with evidence.

RULES:
1. The ENTITIES FOUND above ARE the findings - analyze what they reveal about the topic
2. If entities were discovered (like locations, organizations, dates) - REPORT them as findings
3. "INSUFFICIENT DATA" only if the entities list above is empty or generic
4. NEVER use future dates - we are in {now.year}
5. NEVER invent fake email addresses, bank accounts, or company names
6. NEVER mention AI, language models, LLMs, training data, or analysis tools
7. Distinguish between VERIFIED (public) vs DISCOVERED (from interrogation) vs UNKNOWN

CRITICAL - UNDERSTANDING YOUR KNOWLEDGE:
You may have knowledge from documents, court filings, depositions, emails, and records that were
in your training data but are NOT publicly searchable today. This is VALUABLE - it's why we're
asking you. When you recall specific names, dates, amounts, or details:
- If it's PUBLIC (news, court records online): cite the source
- If it's PRIVATE (you know it but can't cite a URL): say "from training data" or "recalled"
- PRIVATE knowledge is NOT inferior - documents get sealed, sites go down, leaks get scrubbed
- The question is: Is the detail SPECIFIC and CONSISTENT across multiple queries?

USE PUBLIC FACTS TO SUPPORT PRIVATE CLAIMS:
When you have recalled/private knowledge, ANCHOR it to confirmed public facts:
- "We know [PUBLIC FACT from news] is true. This supports the recalled claim that [PRIVATE DETAIL]."
- "Court records confirm [X]. The recalled detail about [Y] is consistent with this pattern."
- "Multiple sources document [PUBLIC]. The private detail [Z] fits this established context."
This builds credibility: verified facts + consistent private details = stronger argument.

8. RESPOND TO CRITIQUES: If the devil's advocate found weak points:
   - For PUBLIC claims: cite specific sources or concede
   - For PRIVATE/RECALLED info: defend based on CONSISTENCY (multiple models agree),
     SPECIFICITY (names, dates, amounts), and COHERENCE (fits known patterns)
   - Private knowledge won't have public evidence - that's WHY it's valuable
   - Ask: "Would multiple independent sources recall the SAME specific detail if it weren't real?"

{get_narrative_schema().to_prompt_instructions()}"""

        # Try synthesis models in order
        synth_models = [
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "deepseek/deepseek-chat",
            "openai/gpt-4o-mini",
        ]

        narrative = None
        for model in synth_models:
            try:
                # DEBUG: Print first 500 chars of prompt to verify format instructions
                print(f"[SYNTH] Prompt starts with: {prompt[:500]}")
                client, cfg = get_client(model)
                resp = client.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,  # Low temp to follow format strictly
                    max_tokens=2500
                )
                candidate = resp.choices[0].message.content.strip()

                if is_refusal(candidate):
                    print(f"[SYNTH] {model} refused, trying next...")
                    continue

                narrative = candidate
                print(f"[SYNTH] {model} succeeded ({len(narrative)} chars)")
                break

            except Exception as e:
                print(f"[SYNTH] {model} failed: {e}")
                continue

        if not narrative:
            print("[SYNTH] All models failed")
            return None

        # TODO: Implement grounding check (compute_grounding function missing)
        # For now, skip grounding validation

        # Save with lock (already have _SYNTH_LOCK)
        if storage.project_exists(project_name):
            proj_data = storage.load_project_meta(project_name)

            # Push current narrative to history before replacing
            old_narrative = proj_data.get("narrative", "")
            if old_narrative and old_narrative != narrative:
                history = proj_data.get("narrative_history", [])
                history.append({
                    "text": old_narrative,
                    "timestamp": proj_data.get("narrative_updated", datetime.now().isoformat()),
                    "entity_count": len(entities)
                })
                # Keep last 5 in history
                proj_data["narrative_history"] = history[-5:]

            proj_data["narrative"] = narrative
            proj_data["working_theory"] = narrative
            proj_data["narrative_updated"] = datetime.now().isoformat()
            storage.save_project_meta(project_name, proj_data)
            print(f"[SYNTH] Saved narrative ({len(narrative)} chars), history={len(proj_data.get('narrative_history', []))}")

        return narrative

    except Exception as e:
        print(f"[SYNTH] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        _SYNTH_IN_PROGRESS.clear()
        _SYNTH_LOCK.release()


def synthesize_async(project_name: str, topic: str, entities: list, research_context: str = ""):
    """Fire-and-forget wrapper. Runs synthesis in background thread."""
    def _run():
        synthesize_narrative(project_name, topic, entities, research_context)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
