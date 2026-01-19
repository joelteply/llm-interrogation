"""
Skeptic Worker - Devil's advocate that challenges findings.

Yin to the interrogator's yang. Both sides see the same evidence
and must prove their case. History is preserved for dialectic context.
"""

import time
from typing import Optional
from collections import Counter

from .base import BaseWorker
from models import SkepticFeedback
from repositories.json_backend import JsonRepository


class SkepticWorker(BaseWorker):
    """
    Provides skeptical analysis of investigation findings.

    Challenges working theory, identifies weak evidence,
    generates counter-questions for falsification.
    """

    def __init__(self, interval: float = 90.0):
        super().__init__(name="SKEPTIC-WORKER", interval=interval)

    def _research_claims(self, topic: str, theory: str) -> str:
        """Research the claims in the theory before critiquing them."""
        from routes.analyze.research.adapters.web_search import WebSearchAdapter

        try:
            searcher = WebSearchAdapter()

            # Extract key entities/claims to search
            # Search for the main topic + key terms from theory
            search_terms = []

            # Get first 500 chars of theory for key terms
            theory_snippet = theory[:500]

            # Build search queries
            queries = [
                f"{topic} evidence",
                f"{topic} documents records",
            ]

            # Look for capitalized org/company names in theory (2+ capitalized words)
            import re
            orgs = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', theory)
            if orgs:
                queries.append(f"{orgs[0]} {topic[:50]}")

            results = []
            for query in queries[:2]:  # Limit to 2 searches
                try:
                    docs = searcher.search(query, max_results=3)
                    for doc in docs[:2]:
                        results.append(f"- [{doc.title}]({doc.url}): {doc.content[:200]}...")
                except Exception as e:
                    print(f"[SKEPTIC] Search failed for '{query}': {e}")

            if results:
                return "SKEPTIC'S OWN RESEARCH FOUND:\n" + "\n".join(results)
            return "No additional research found."

        except Exception as e:
            print(f"[SKEPTIC] Research failed: {e}")
            return "Research unavailable."

    def _do_work(self, project_name: str) -> int:
        """Analyze project with skepticism - WITH RESEARCH."""
        from routes import project_storage as storage
        from config import get_client
        from routes.analyze.research.adapters.web_search import WebSearchAdapter

        if not storage.project_exists(project_name):
            return 0

        proj = storage.load_project_meta(project_name)
        topic = proj.get("topic", "")
        working_theory = proj.get("working_theory", "") or proj.get("narrative", "")
        research_context = proj.get("research_context", "")

        if not working_theory:
            return 0

        # SKEPTIC MUST DO ITS OWN RESEARCH before dismissing claims
        print(f"[{self.name}] Researching claims before critique...")
        skeptic_research = self._research_claims(topic, working_theory)

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

        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%B %d, %Y at %I:%M %p")

        prompt = f"""You are a skeptical analyst in a DEBATE. The theory writer will see your critique and can call you out if you're lazy or wrong.

TODAY'S DATE: {date_str}. Events before today are PAST TENSE and can be verified.

INVESTIGATION TOPIC:
{topic}

CURRENT WORKING THEORY:
{working_theory}

KEY ENTITIES FOUND (with mention counts and sample context):
{chr(10).join(entity_evidence)}

RESEARCH GATHERED BY INTERROGATOR:
{research_context[:1500] if research_context else "No external research yet."}

{skeptic_research}

---

CRITICAL: READ THE RESEARCH ABOVE BEFORE RESPONDING.

YOU WILL LOOK LIKE AN IDIOT IF:
- You say "no verifiable source" when SOURCES ARE RIGHT ABOVE in the research section
- You dismiss claims that are actually documented in news/court records
- You say something is "vague" when specific names, dates, and amounts are provided
- You ignore the research section entirely and make lazy blanket dismissals

The theory writer WILL call you out: "You didn't look it up. It's documented. Prove your dismissal."

YOUR JOB:
1. FIRST: Quote what the research section found. Did it confirm or contradict?
2. THEN: If research CONFIRMS the claim, acknowledge it. Find a DIFFERENT weakness.
3. If you say "unsupported" - you MUST cite what you searched that came up empty
4. Past tense claims (2023-2025) are VERIFIABLE. Did you verify them?

WHAT MAKES A VALID CRITIQUE:
- "The research found X, but this doesn't prove Y because..."
- "While [entity] exists (confirmed), the specific $X amount is not in any source"
- "The timeline is impossible because [specific contradiction]"
- "This contradicts known fact X from source Y"

WHAT MAKES YOU LOOK LAZY/DUMB:
- "No verifiable source" (when sources exist above)
- "Vague language" (when specific details are given)
- "Cannot be confirmed" (without saying what you searched)
- Ignoring the research section entirely

Respond in this exact format:

CONFIRMED PUBLIC FACTS I FOUND (list each one):
1. [Fact] - Source: [where found]
2. [Fact] - Source: [outlet/document]
3. [List ALL public facts your research confirmed - be thorough]

WHAT_I_FOUND: [QUOTE from the research above. What did it say? Did it CONFIRM or CONTRADICT the theory? Be specific. If research confirms the claims, SAY SO.]

WEAKEST_LINK: [The single least-supported claim. If your research CONFIRMED most claims, find something ELSE to challenge - a logical leap, missing connection, or genuinely unverified detail.]

ALTERNATIVE_EXPLANATION: [What ELSE could explain this? Not "it might not be true" - an actual alternative theory.]

COUNTER_QUESTIONS: [3 questions that would DISPROVE the theory IF answered a certain way. These should be falsifiable.]

CONFIDENCE_ASSESSMENT: [If research confirms key claims = MEDIUM or HIGH. Only LOW if research contradicts or finds nothing.]"""

        response = client.chat.completions.create(
            model=cfg['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1500
        )

        content = response.choices[0].message.content.strip()
        parsed = self._parse_skeptic_response(content)

        # Build SkepticFeedback model with theory snapshot for dialectic context
        from datetime import datetime
        feedback = SkepticFeedback(
            weakest_link=parsed.get("weakest_link"),
            what_i_found=parsed.get("what_i_found"),  # Skeptic's own research
            alternative_explanation=parsed.get("alternative_explanation"),
            circular_evidence=parsed.get("circular_evidence", []),
            counter_questions=parsed.get("counter_questions", []),
            missing_research=parsed.get("missing_research"),
            confidence=parsed.get("confidence", "LOW"),
            updated_at=datetime.now(),  # Set explicitly NOW
            theory_snapshot=working_theory,  # What theory was challenged
            raw=content
        )

        # Append to history for dialectic - both sides see the debate
        repo = JsonRepository()
        repo.projects.append_skeptic_history(project_name, feedback)

        # Also save as current feedback (quick access)
        proj["skeptic_feedback"] = feedback.model_dump(mode="json")
        storage.save_project_meta(project_name, proj)

        print(f"[{self.name}] Done - confidence: {feedback.confidence}")

        return len(feedback.counter_questions)

    def _parse_skeptic_response(self, content: str) -> dict:
        """Parse the structured skeptic response."""
        feedback = {
            "confirmed_facts": [],
            "what_i_found": "",
            "weakest_link": "",
            "alternative_explanation": "",
            "circular_evidence": [],
            "counter_questions": [],
            "missing_research": "",
            "confidence": "MEDIUM"
        }

        key_map = {
            "CONFIRMED PUBLIC FACTS": "confirmed_facts",
            "CONFIRMED_PUBLIC_FACTS": "confirmed_facts",
            "WEAKEST_LINK": "weakest_link",
            "WHAT_I_FOUND": "what_i_found",
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
                    if current_key in ["circular_evidence", "counter_questions", "confirmed_facts"]:
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
            if current_key in ["circular_evidence", "counter_questions", "confirmed_facts"]:
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
