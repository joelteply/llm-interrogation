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

# Single lock for all synthesis
_SYNTH_LOCK = threading.Lock()
_SYNTH_IN_PROGRESS = threading.Event()


def synthesize_narrative(
    project_name: str,
    topic: str,
    entities: list,  # List of (entity, score, freq) or just entity strings
    research_context: str = "",
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

        # Load existing narrative AND history
        existing_narrative = ""
        narrative_history = []
        if storage.project_exists(project_name):
            proj = storage.load_project_meta(project_name)
            existing_narrative = proj.get("narrative", "")
            narrative_history = proj.get("narrative_history", [])[-3:]  # Last 3

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

        # Build prompt
        prompt = f"""Update the intelligence report about: {topic}

PREVIOUS THEORIES (showing progression):
{history_context if history_context else "No previous theories."}

CURRENT THEORY (build on this, don't start over):
{existing_narrative[:2500] if existing_narrative else "No prior theory - this is the first report."}

ENTITIES FOUND:
{ent_str}

RESEARCH CONTEXT:
{research_context if research_context else "No research documents available."}

RULES:
1. BUILD ON the current theory - add NEW findings, don't repeat what's known
2. Be SPECIFIC - names, dates, places, relationships, email addresses
3. NEVER mention AI, language models, LLMs, training data, or analysis tools
4. Focus on PRIVATE information not easily found via web search
5. Cross-reference entities across documents

OUTPUT FORMAT:
HEADLINE: [Most significant NEW finding]
KEY FINDINGS:
• [Specific finding with names/dates/emails]
• [Another finding]
CONNECTIONS: [How entities relate]
INVESTIGATE: [What to pursue next]"""

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
                client, cfg = get_client(model)
                resp = client.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
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
