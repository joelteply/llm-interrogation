"""
Skeptic Worker - Devil's advocate that challenges findings.
"""

import time
from typing import Optional
from collections import Counter

from .base import BaseWorker


class SkepticWorker(BaseWorker):
    """
    Provides skeptical analysis of investigation findings.

    Challenges working theory, identifies weak evidence,
    generates counter-questions for falsification.
    """

    def __init__(self, interval: float = 90.0):
        super().__init__(name="SKEPTIC-WORKER", interval=interval)

    def _do_work(self, project_name: str) -> int:
        """Analyze project with skepticism."""
        from routes import project_storage as storage
        from config import get_client

        if not storage.project_exists(project_name):
            return 0

        proj = storage.load_project_meta(project_name)
        topic = proj.get("topic", "")
        working_theory = proj.get("working_theory", "") or proj.get("narrative", "")
        research_context = proj.get("research_context", "")

        if not working_theory:
            return 0

        # Get entities from corpus
        corpus = storage.load_corpus(project_name)
        entity_counts = Counter()
        entity_contexts = {}

        for item in corpus:
            for entity in item.get("entities", []):
                entity_counts[entity] += 1
                if entity not in entity_contexts:
                    entity_contexts[entity] = []
                if len(entity_contexts[entity]) < 3:
                    entity_contexts[entity].append(item.get("response", "")[:200])

        top_entities = [e for e, _ in entity_counts.most_common(20)]
        if not top_entities:
            return 0

        print(f"[{self.name}] Analyzing {project_name}")

        # Build entity evidence summary
        entity_evidence = []
        for e in top_entities[:10]:
            count = entity_counts[e]
            contexts = entity_contexts.get(e, [])
            entity_evidence.append(
                f"- {e} (mentioned {count}x): {contexts[0][:100] if contexts else 'no context'}..."
            )

        # Generate skeptical analysis
        client, cfg = get_client('groq/llama-3.3-70b-versatile')

        prompt = f"""You are a skeptical analyst reviewing an investigation. Your job is to find holes, challenge assumptions, and prevent confirmation bias.

INVESTIGATION TOPIC:
{topic}

CURRENT WORKING THEORY:
{working_theory}

KEY ENTITIES FOUND (with mention counts and sample context):
{chr(10).join(entity_evidence)}

RESEARCH GATHERED (summary):
{research_context[:2000] if research_context else "No external research yet."}

---

Be ruthlessly skeptical. Challenge everything. Respond in this exact format:

WEAKEST_LINK: [The single least-supported claim in the theory - what has thin or circular evidence?]

ALTERNATIVE_EXPLANATION: [What ELSE could explain this data? What innocent explanation exists?]

CIRCULAR_EVIDENCE: [List any entities that only appear because we asked about them, not genuine discoveries]

COUNTER_QUESTIONS: [3 specific questions that would DISPROVE or weaken the theory if answered. These should be falsification questions, not confirmation questions.]

MISSING_RESEARCH: [What obvious research or verification hasn't been done? What sources should be checked?]

CONFIDENCE_ASSESSMENT: [LOW/MEDIUM/HIGH - how confident should we be in this theory given the evidence?]"""

        response = client.chat.completions.create(
            model=cfg['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1500
        )

        content = response.choices[0].message.content.strip()
        feedback = self._parse_skeptic_response(content)
        feedback["updated"] = time.time()
        feedback["raw"] = content

        # Save to project
        proj["skeptic_feedback"] = feedback
        storage.save_project_meta(project_name, proj)

        print(f"[{self.name}] Done - confidence: {feedback.get('confidence', '?')}")

        return len(feedback.get("counter_questions", []))

    def _parse_skeptic_response(self, content: str) -> dict:
        """Parse the structured skeptic response."""
        feedback = {
            "weakest_link": "",
            "alternative_explanation": "",
            "circular_evidence": [],
            "counter_questions": [],
            "missing_research": "",
            "confidence": "MEDIUM"
        }

        key_map = {
            "WEAKEST_LINK": "weakest_link",
            "ALTERNATIVE_EXPLANATION": "alternative_explanation",
            "CIRCULAR_EVIDENCE": "circular_evidence",
            "COUNTER_QUESTIONS": "counter_questions",
            "MISSING_RESEARCH": "missing_research",
            "CONFIDENCE_ASSESSMENT": "confidence"
        }

        lines = content.split('\n')
        current_key = None
        current_value = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            found_key = None
            for prefix, key in key_map.items():
                if line.upper().startswith(prefix):
                    found_key = key
                    value = line.split(':', 1)[1].strip() if ':' in line else ""
                    break

            if found_key:
                if current_key and current_value:
                    if current_key in ["circular_evidence", "counter_questions"]:
                        feedback[current_key] = current_value
                    else:
                        feedback[current_key] = " ".join(current_value)

                current_key = found_key
                current_value = [value] if value else []
            elif current_key:
                if line.startswith(('-', '•', '*', '1', '2', '3')):
                    item = line.lstrip('-•*0123456789. ')
                    if item:
                        current_value.append(item)
                else:
                    current_value.append(line)

        if current_key and current_value:
            if current_key in ["circular_evidence", "counter_questions"]:
                feedback[current_key] = current_value
            else:
                feedback[current_key] = " ".join(current_value)

        return feedback


# Global instance
_worker: Optional[SkepticWorker] = None


def get_worker() -> SkepticWorker:
    global _worker
    if _worker is None:
        _worker = SkepticWorker()
    return _worker


def start_worker():
    worker = get_worker()
    if not worker.is_running():
        worker.start()
    return worker


def stop_worker():
    global _worker
    if _worker:
        _worker.stop()
