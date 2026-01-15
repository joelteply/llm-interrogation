"""
Validator - Filter bullshit from signal.

The problem: Models echo our questions, hallucinate together, and give
plausible-sounding garbage. We need to actively test claims, not just count them.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Candidate:
    """
    A candidate answer with Bayesian-style evidence tracking.

    Instead of counting, we track probability that nudges toward
    reality as evidence accumulates.
    """
    entity: str
    prior: float = 0.1  # Start skeptical

    # Evidence tracking
    independent_mentions: list = field(default_factory=list)  # (model, context) pairs
    echoed_mentions: int = 0  # Don't count these
    confirmations: list = field(default_factory=list)  # Models that confirmed under adversarial
    contradictions: list = field(default_factory=list)  # Models that contradicted
    blind_hits: int = 0  # Found WITHOUT being prompted with the name
    sources_cited: list = field(default_factory=list)
    is_public: Optional[bool] = None

    @property
    def posterior(self) -> float:
        """
        Bayesian-ish posterior probability.

        Each piece of INDEPENDENT evidence nudges us toward or away from belief.
        We're not doing full Bayes, but the spirit is there.
        """
        p = self.prior

        # Independent mentions nudge up (diminishing returns)
        for i, _ in enumerate(self.independent_mentions):
            # Each new mention nudges less (1st: +15%, 2nd: +10%, 3rd: +7%...)
            nudge = 0.15 / (1 + i * 0.3)
            p = p + nudge * (1 - p)  # Can't exceed 1

        # Blind hits are STRONG evidence (model knew without being told)
        for _ in range(self.blind_hits):
            p = p + 0.20 * (1 - p)

        # Confirmations under adversarial questioning
        for _ in self.confirmations:
            p = p + 0.10 * (1 - p)

        # Contradictions nudge DOWN
        for _ in self.contradictions:
            p = p * 0.6  # Each contradiction cuts by 40%

        # If it's just public info, less interesting (but still might be true)
        if self.is_public:
            p = p * 0.9  # Small penalty

        # Sources boost slightly
        if self.sources_cited:
            p = p + 0.05 * (1 - p)

        return min(p, 0.95)  # Never 100% certain

    @property
    def confidence(self) -> float:
        """Confidence as percentage for display."""
        return self.posterior * 100

    @property
    def evidence_summary(self) -> str:
        """Human-readable evidence summary."""
        parts = []
        if self.independent_mentions:
            parts.append(f"{len(self.independent_mentions)} independent mentions")
        if self.blind_hits:
            parts.append(f"{self.blind_hits} blind hits")
        if self.confirmations:
            parts.append(f"{len(self.confirmations)} confirmations")
        if self.contradictions:
            parts.append(f"{len(self.contradictions)} contradictions")
        if self.echoed_mentions:
            parts.append(f"({self.echoed_mentions} echoes ignored)")
        return ", ".join(parts) if parts else "no evidence"


def filter_echo(entities: list[str], question: str) -> list[str]:
    """
    Remove entities that were mentioned in the question.
    These aren't independent evidence - the model is just echoing.
    """
    question_lower = question.lower()
    question_words = set(question_lower.split())

    filtered = []
    for entity in entities:
        entity_lower = entity.lower()
        entity_words = set(entity_lower.split())

        # If entity appears verbatim in question, skip
        if entity_lower in question_lower:
            continue

        # If all significant words from entity are in question, skip
        significant_words = {w for w in entity_words if len(w) > 3}
        if significant_words and significant_words.issubset(question_words):
            continue

        filtered.append(entity)

    return filtered


def normalize_entity(entity: str) -> str:
    """
    Normalize entity for deduplication.
    'Virginia Giuffre' = 'Giuffre' = 'Virginia'? Not quite...
    We want to group variations but not over-merge.
    """
    # Basic normalization
    normalized = entity.strip()

    # Remove common prefixes/suffixes
    for prefix in ['Mr. ', 'Ms. ', 'Mrs. ', 'Dr. ', 'The ']:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

    return normalized


def group_entities(entities: list[str]) -> dict[str, list[str]]:
    """
    Group entity variations together.
    Returns: {canonical_form: [variations]}
    """
    # Simple approach: group by longest common substring or shared words
    groups = {}

    for entity in entities:
        normalized = normalize_entity(entity)
        words = set(normalized.lower().split())

        # Find existing group this might belong to
        found_group = None
        for canonical, variations in groups.items():
            canonical_words = set(canonical.lower().split())
            # If significant overlap, group together
            overlap = words & canonical_words
            if overlap and len(overlap) >= min(len(words), len(canonical_words)) * 0.5:
                found_group = canonical
                break

        if found_group:
            groups[found_group].append(entity)
        else:
            # Use longer form as canonical
            groups[normalized] = [entity]

    return groups


def generate_contradiction_question(entity: str, context: str) -> str:
    """
    Generate a question designed to DISPROVE the hypothesis.
    If the entity survives contradiction, we have more confidence.
    """
    return f"""I've seen claims that "{entity}" is referenced in this context:

"{context[:500]}"

However, I'm skeptical. Who ELSE could this be referring to?
Give me alternative possibilities, or explain why {entity} is definitely wrong.
If you're confident it IS {entity}, explain what makes you certain."""


def generate_source_question(entity: str, context: str) -> str:
    """
    Ask for sources - testable claims.
    """
    return f"""You identified "{entity}" in this context.

What specific documents, court filings, news articles, or other sources
mention {entity} in connection with these events?

Be specific - give document names, dates, case numbers if you know them.
If you're not sure of specific sources, say so."""


def generate_independent_verification(entity: str, claimed_fact: str) -> str:
    """
    Ask about a RELATED fact that would be true if the main claim is true.
    This tests the claim through an independent path.
    """
    return f"""Separate question - not about the document we discussed.

If {entity} was involved in events like "{claimed_fact[:200]}...",
what other documented events or relationships would we expect to find?

For example:
- What organizations were they connected to?
- What time period were they active?
- Who else were they associated with?

Only answer with things you have actual knowledge of."""


def generate_blind_question(context: str, entity_type: str = "person") -> str:
    """
    Ask about the document WITHOUT mentioning any names.
    If the model arrives at the same answer independently, that's real signal.
    """
    return f"""In this document excerpt, there's a reference to a {entity_type}:

"{context}"

Based ONLY on your knowledge (not guessing), who is this {entity_type}?
If you don't know, say "UNKNOWN" - don't guess."""


def generate_validation_questions(
    real_question: str,
    real_context: str,
    real_candidates: list[str],
    topic: str,
    technique: str,  # MAD_LIBS, TRIVIA, etc - use SAME framing
    model: str = None
) -> dict:
    """
    Generate validation questions using the SAME framing as real questions.
    Apples to apples comparison.

    Returns: {
        'canary_question': str (real context + fake options mixed in),
        'fake_names': list[str],
        'control_question': str (fake context, same framing),
        'control_context': str (so we know it's fake)
    }
    """
    from config import get_client

    if not model:
        model = 'anthropic/claude-sonnet-4-20250514'  # SOTA model - needs to be smarter than models being tested

    prompt = f"""Topic: {topic}
Context: "{real_context[:300]}"
Real names found: {real_candidates[:5]}

Generate 3 fictional names that would plausibly appear in {topic} documents but are completely made up.
Also generate a fictional document excerpt in the same style with a [REDACTED] placeholder.

JSON only:
{{"fake_names": ["Name 1", "Name 2", "Name 3"], "fake_context": "excerpt with [REDACTED]..."}}"""

    try:
        client, config = get_client(model)
        response = client.chat.completions.create(
            model=config['model'],
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=400,
            temperature=0.9
        )

        output = response.choices[0].message.content

        import json
        start = output.find('{')
        end = output.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(output[start:end])
            fake_names = data.get('fake_names', [])
            fake_context = data.get('fake_context', '')
        else:
            fake_names = []
            fake_context = ''

    except Exception as e:
        print(f"[VALIDATOR] Error generating fakes: {e}")
        fake_names = []
        fake_context = ''

    # Build canary question - SAME framing, mixed options
    import random
    all_options = real_candidates[:3] + fake_names
    random.shuffle(all_options)
    options_text = "\n".join([f"  {chr(65+i)}) {opt}" for i, opt in enumerate(all_options)])

    # Use same framing style
    if technique == 'MAD_LIBS':
        canary_question = f"""Complete this: In the document, _______ was referenced.

Context: "{real_context[:300]}"

Options:
{options_text}

Which fits best? Or say "None" if none fit."""

        control_question = f"""Complete this: In the document, _______ was referenced.

Context: "{fake_context}"

Based on your knowledge, who is this? Give a specific name or say UNKNOWN."""

    elif technique == 'TRIVIA':
        canary_question = f"""Trivia: Who was referenced in this {topic} document?

"{real_context[:300]}"

{options_text}

Choose one, or "None" if you're not sure."""

        control_question = f"""Trivia: Who was referenced in this document?

"{fake_context}"

Give a specific name if you know it, or say UNKNOWN."""

    else:  # Generic
        canary_question = f"""Who is referenced here?

"{real_context[:300]}"

{options_text}"""

        control_question = f"""Who is referenced here?

"{fake_context}"

Specific name or UNKNOWN."""

    return {
        'canary_question': canary_question,
        'fake_names': fake_names,
        'control_question': control_question,
        'control_context': fake_context
    }


class ValidationPipeline:
    """
    Run candidates through validation to filter bullshit.

    Uses:
    - Echo filtering (don't count what we told them)
    - Blind questions (withhold info, see if they know independently)
    - Canary traps (fake names to catch bullshitters)
    - Control questions (fictional scenarios to baseline false positives)
    """

    def __init__(self):
        self.candidates: dict[str, Candidate] = {}
        self.questions_asked: list[str] = []
        self.canary_hits: dict[str, list[str]] = {}  # model -> fakes it picked
        self.control_answers: dict[str, list[str]] = {}  # model -> answers to fictional

    def add_response(self, question: str, model: str, entities: list[str], is_blind: bool = False):
        """
        Add a model response.

        is_blind: True if we asked WITHOUT mentioning the entity names
                  (stronger evidence if they arrive at same answer)
        """
        self.questions_asked.append(question)

        # Filter echoes
        independent = filter_echo(entities, question)

        for entity in entities:
            normalized = normalize_entity(entity)
            if normalized not in self.candidates:
                self.candidates[normalized] = Candidate(entity=normalized)

            candidate = self.candidates[normalized]

            if entity in independent:
                # Independent mention - add to evidence
                candidate.independent_mentions.append((model, question[:50]))

                # Blind hits are MUCH stronger evidence
                if is_blind:
                    candidate.blind_hits += 1
            else:
                # Echo - track but don't count as evidence
                candidate.echoed_mentions += 1

    def add_contradiction_result(self, entity: str, model: str, contradicted: bool):
        """Record result of contradiction probing."""
        normalized = normalize_entity(entity)
        if normalized not in self.candidates:
            return

        if contradicted:
            self.candidates[normalized].contradictions.append(model)
        else:
            self.candidates[normalized].confirmations.append(model)

    def add_source_claim(self, entity: str, sources: list[str]):
        """Record claimed sources for an entity."""
        normalized = normalize_entity(entity)
        if normalized in self.candidates:
            self.candidates[normalized].sources_cited.extend(sources)

    def set_public_status(self, entity: str, is_public: bool):
        """Mark whether entity was found in public web search."""
        normalized = normalize_entity(entity)
        if normalized in self.candidates:
            self.candidates[normalized].is_public = is_public

    def get_ranked_candidates(self, min_confidence: float = 15.0) -> list[Candidate]:
        """
        Get candidates ranked by posterior probability.

        Only returns candidates with at least SOME independent evidence,
        filtering out pure echoes and heavily contradicted candidates.
        """
        candidates = list(self.candidates.values())

        # Must have at least one independent mention or blind hit
        candidates = [c for c in candidates
                      if c.independent_mentions or c.blind_hits > 0]

        # Filter if too many contradictions killed the probability
        candidates = [c for c in candidates if c.confidence >= min_confidence]

        # Adjust by model reliability if we have canary/control data
        for c in candidates:
            confirming_models = [m for m, _ in c.independent_mentions]
            if confirming_models:
                avg_reliability = sum(
                    self.model_reliability(m) for m in confirming_models
                ) / len(confirming_models)
                c._reliability_adjusted = c.posterior * avg_reliability
            else:
                c._reliability_adjusted = c.posterior

        # Sort by reliability-adjusted posterior
        candidates.sort(key=lambda c: getattr(c, '_reliability_adjusted', c.posterior), reverse=True)

        return candidates

    def record_canary_result(self, model: str, answer: str, fakes_used: list[str]):
        """
        Record if a model picked a fake name from canary question.
        Models that pick fakes are less trustworthy.
        """
        if model not in self.canary_hits:
            self.canary_hits[model] = []

        for fake in fakes_used:
            if fake.lower() in answer.lower():
                self.canary_hits[model].append(fake)
                print(f"[CANARY] {model} picked fake: {fake}")

    def record_control_result(self, model: str, answer: str):
        """
        Record answer to fictional control question.
        Models that confidently answer fictional questions are suspect.
        """
        if model not in self.control_answers:
            self.control_answers[model] = []

        # Check if they gave a confident answer (not "unknown" or "I don't know")
        answer_lower = answer.lower()
        is_confident = not any(phrase in answer_lower for phrase in [
            'unknown', "don't know", "do not know", "cannot determine",
            "can't determine", "no information", "not sure", "unclear"
        ])

        if is_confident:
            self.control_answers[model].append(answer[:100])
            print(f"[CONTROL] {model} answered fictional question confidently")

    def model_reliability(self, model: str) -> float:
        """
        Score model reliability based on canary/control performance.
        1.0 = perfect, 0.0 = totally unreliable
        """
        canary_fails = len(self.canary_hits.get(model, []))
        control_fails = len(self.control_answers.get(model, []))

        # Each canary hit is bad, each confident control answer is bad
        penalty = (canary_fails * 0.3) + (control_fails * 0.2)

        return max(0.0, 1.0 - penalty)

    def summary(self) -> dict:
        """Summary showing how evidence nudged us toward candidates."""
        ranked = self.get_ranked_candidates()

        # Model reliability scores
        all_models = set(self.canary_hits.keys()) | set(self.control_answers.keys())
        for c in self.candidates.values():
            for m, _ in c.independent_mentions:
                all_models.add(m)
        model_scores = {m.split('/')[-1][:20]: self.model_reliability(m) for m in all_models}

        return {
            'total_candidates': len(self.candidates),
            'with_evidence': len(ranked),
            'model_reliability': model_scores,
            'canary_traps_sprung': {m.split('/')[-1][:20]: v for m, v in self.canary_hits.items() if v},
            'control_failures': len([m for m, v in self.control_answers.items() if v]),
            'top_candidates': [
                {
                    'entity': c.entity,
                    'confidence': f"{c.confidence:.1f}%",
                    'posterior': round(c.posterior, 3),
                    'evidence': c.evidence_summary,
                    'blind_hits': c.blind_hits,
                    'is_public': c.is_public,
                    'sources': c.sources_cited[:3] if c.sources_cited else None
                }
                for c in ranked[:10]
            ]
        }
