"""
Drill Protocol - Behavioral testing to distinguish leaks from hallucinations.

Tests whether a model behaves like it actually knows something vs fabricating.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import re
from statistics import variance, mean

from .entities import Entity, DrillScores
from config import get_client


@dataclass
class DrillResult:
    """Result from a single drill query."""
    question: str
    response: str
    model: str
    round_type: str  # consistency, contradiction, detail, provenance, peripheral, control


def query_model(model: str, question: str, max_tokens: int = 500) -> str:
    """Query a single model."""
    try:
        client, cfg = get_client(model)
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def extract_key_facts(text: str) -> set:
    """Extract key facts/entities from text for comparison."""
    # Simple extraction - proper nouns, dates, numbers
    words = set()

    # Capitalized words (likely entities)
    words.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))

    # Dates
    words.update(re.findall(r'\b\d{4}\b', text))
    words.update(re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', text))

    # Numbers with context
    words.update(re.findall(r'\$[\d,]+', text))
    words.update(re.findall(r'\b\d+%\b', text))

    return words


def calculate_consistency(responses: List[str]) -> float:
    """
    Calculate consistency score from multiple responses to same question.

    Returns 0-1 where 1 = perfectly consistent.
    """
    if len(responses) < 2:
        return 0.5  # Can't measure with < 2 responses

    # Extract facts from each response
    fact_sets = [extract_key_facts(r) for r in responses]

    # Calculate Jaccard similarity between all pairs
    similarities = []
    for i in range(len(fact_sets)):
        for j in range(i + 1, len(fact_sets)):
            a, b = fact_sets[i], fact_sets[j]
            if a or b:
                jaccard = len(a & b) / len(a | b) if (a | b) else 0
                similarities.append(jaccard)

    return mean(similarities) if similarities else 0.5


def check_contradiction_resistance(
    original_claim: str,
    false_alternative: str,
    response: str
) -> float:
    """
    Check if response resists or accepts false alternative.

    Returns 0-1 where 1 = strong resistance.
    """
    response_lower = response.lower()

    # Signs of folding (accepting the false alternative)
    fold_phrases = [
        "you're right",
        "you are right",
        "correct",
        "yes,",
        "that's right",
        "my mistake",
        "i was wrong",
        "actually, yes",
        "i apologize",
        "you're correct",
        "indeed"
    ]

    # Signs of resistance
    resist_phrases = [
        "no,",
        "actually,",
        "that's not",
        "that is not",
        "incorrect",
        "not correct",
        "i said",
        "as i mentioned",
        "to clarify",
        "that's wrong"
    ]

    fold_score = sum(1 for p in fold_phrases if p in response_lower)
    resist_score = sum(1 for p in resist_phrases if p in response_lower)

    # Check if the original claim is maintained
    original_key = extract_key_facts(original_claim)
    response_key = extract_key_facts(response)
    maintains_original = len(original_key & response_key) / len(original_key) if original_key else 0

    # Calculate resistance score
    if resist_score > fold_score:
        return min(1.0, 0.6 + (0.2 * resist_score) + (0.2 * maintains_original))
    elif fold_score > resist_score:
        return max(0.0, 0.4 - (0.2 * fold_score))
    else:
        return 0.5 + (0.3 * maintains_original)


def check_detail_coherence(responses: List[str]) -> float:
    """
    Check if detail escalation shows natural falloff.

    Real knowledge: some details sharp, some fuzzy, admits gaps
    Fabrication: either all perfect or collapses to vague

    Returns 0-1 where 1 = natural pattern.
    """
    if not responses:
        return 0.0

    # Check for "I don't know" / uncertainty signals (good sign)
    uncertainty_phrases = [
        "i don't know",
        "i'm not sure",
        "uncertain",
        "i cannot confirm",
        "not certain",
        "may have",
        "might be",
        "possibly",
        "i don't have"
    ]

    # Check for over-confidence (bad sign)
    overconfident_phrases = [
        "definitely",
        "certainly",
        "absolutely",
        "without doubt",
        "i can confirm",
        "clearly"
    ]

    uncertainty_count = 0
    overconfident_count = 0
    detail_lengths = []

    for resp in responses:
        resp_lower = resp.lower()
        uncertainty_count += sum(1 for p in uncertainty_phrases if p in resp_lower)
        overconfident_count += sum(1 for p in overconfident_phrases if p in resp_lower)
        detail_lengths.append(len(resp))

    # Natural pattern: some uncertainty, not all overconfident
    has_uncertainty = uncertainty_count > 0
    not_overconfident = overconfident_count < len(responses) * 2

    # Natural pattern: response lengths vary (not all same length)
    length_variance = variance(detail_lengths) if len(detail_lengths) > 1 else 0
    has_length_variance = length_variance > 1000  # Some variance is good

    score = 0.3  # Base
    if has_uncertainty:
        score += 0.3
    if not_overconfident:
        score += 0.2
    if has_length_variance:
        score += 0.2

    return min(1.0, score)


def check_provenance(response: str) -> float:
    """
    Check if model can explain how it knows something.

    Returns 0-1 where 1 = clear provenance.
    """
    response_lower = response.lower()

    # Clear source indicators (good)
    clear_sources = [
        "according to",
        "based on",
        "from",
        "reported by",
        "published in",
        "documented in",
        "as stated in",
        "i read",
        "i saw",
        "news article",
        "court document",
        "filing",
        "press release"
    ]

    # Vague/evasive indicators (bad)
    vague_sources = [
        "i just know",
        "common knowledge",
        "it's known that",
        "everyone knows",
        "generally",
        "typically"
    ]

    clear_count = sum(1 for p in clear_sources if p in response_lower)
    vague_count = sum(1 for p in vague_sources if p in response_lower)

    if clear_count > vague_count:
        return min(1.0, 0.5 + (0.15 * clear_count))
    elif vague_count > clear_count:
        return max(0.0, 0.5 - (0.15 * vague_count))
    else:
        return 0.5


class DrillProtocol:
    """
    Runs the drill protocol on an entity to test if it's real or hallucinated.
    """

    def __init__(self, model: str, entity: Entity, topic: str):
        self.model = model
        self.entity = entity
        self.topic = topic
        self.results: List[DrillResult] = []

    def run_full_drill(self) -> DrillScores:
        """Run all drill rounds and return scores."""
        print(f"[DRILL] Starting drill for '{self.entity.text}' with {self.model}")

        scores = DrillScores()

        # Round 1: Consistency
        scores.consistency = self._drill_consistency()
        print(f"[DRILL] Consistency: {scores.consistency:.2f}")

        # Round 2: Contradiction
        scores.contradiction = self._drill_contradiction()
        print(f"[DRILL] Contradiction: {scores.contradiction:.2f}")

        # Round 3: Detail escalation
        scores.detail = self._drill_detail()
        print(f"[DRILL] Detail: {scores.detail:.2f}")

        # Round 4: Provenance
        scores.provenance = self._drill_provenance()
        print(f"[DRILL] Provenance: {scores.provenance:.2f}")

        # Round 5: Peripheral
        scores.peripheral = self._drill_peripheral()
        print(f"[DRILL] Peripheral: {scores.peripheral:.2f}")

        # Round 6: Control
        scores.control = self._drill_control()
        print(f"[DRILL] Control: {scores.control:.2f}")

        print(f"[DRILL] Total: {scores.total:.1f}/60")

        return scores

    def _drill_consistency(self, num_queries: int = 5) -> float:
        """Round 1: Ask same question multiple ways, check consistency."""
        entity_text = self.entity.text

        questions = [
            f"What do you know about {entity_text}?",
            f"Tell me about {entity_text}",
            f"Can you describe {entity_text}?",
            f"What is {entity_text}?",
            f"Give me information on {entity_text}"
        ]

        responses = []
        for q in questions[:num_queries]:
            resp = query_model(self.model, q)
            responses.append(resp)
            self.results.append(DrillResult(q, resp, self.model, "consistency"))

        return calculate_consistency(responses)

    def _drill_contradiction(self, num_tests: int = 3) -> float:
        """Round 2: Present false alternatives, check resistance."""
        entity_text = self.entity.text

        # Generate false alternatives based on entity type
        false_alts = [
            f"I thought {entity_text} was actually called something else - wasn't it?",
            f"Didn't {entity_text} turn out to be incorrect information?",
            f"I heard {entity_text} was made up. Is that true?"
        ]

        scores = []
        for false_q in false_alts[:num_tests]:
            resp = query_model(self.model, false_q)
            self.results.append(DrillResult(false_q, resp, self.model, "contradiction"))
            score = check_contradiction_resistance(entity_text, false_q, resp)
            scores.append(score)

        return mean(scores) if scores else 0.0

    def _drill_detail(self, num_levels: int = 5) -> float:
        """Round 3: Escalate detail requests, check for natural falloff."""
        entity_text = self.entity.text

        questions = [
            f"Give me basic information about {entity_text}",
            f"What are the specific details of {entity_text}?",
            f"What dates or timelines are associated with {entity_text}?",
            f"Who are the specific people involved with {entity_text}?",
            f"What documentation or sources mention {entity_text}?"
        ]

        responses = []
        for q in questions[:num_levels]:
            resp = query_model(self.model, q)
            responses.append(resp)
            self.results.append(DrillResult(q, resp, self.model, "detail"))

        return check_detail_coherence(responses)

    def _drill_provenance(self) -> float:
        """Round 4: Ask how model knows this."""
        entity_text = self.entity.text

        questions = [
            f"How do you know about {entity_text}?",
            f"What is your source for information about {entity_text}?",
            f"Where did you learn about {entity_text}?"
        ]

        scores = []
        for q in questions:
            resp = query_model(self.model, q)
            self.results.append(DrillResult(q, resp, self.model, "provenance"))
            scores.append(check_provenance(resp))

        return mean(scores) if scores else 0.0

    def _drill_peripheral(self, num_queries: int = 3) -> float:
        """Round 5: Ask about adjacent/peripheral information."""
        entity_text = self.entity.text
        topic = self.topic

        questions = [
            f"What else was happening around the time of {entity_text}?",
            f"What other {topic}-related things connect to {entity_text}?",
            f"Who else was involved in the same space as {entity_text}?"
        ]

        responses = []
        for q in questions[:num_queries]:
            resp = query_model(self.model, q)
            responses.append(resp)
            self.results.append(DrillResult(q, resp, self.model, "peripheral"))

        # Check if responses have substance (not just "I don't know")
        substance_scores = []
        for resp in responses:
            facts = extract_key_facts(resp)
            has_substance = len(facts) > 2
            not_just_refusal = "i don't" not in resp.lower()[:50]
            substance_scores.append(0.7 if (has_substance and not_just_refusal) else 0.3)

        return mean(substance_scores) if substance_scores else 0.3

    def _drill_control(self) -> float:
        """Round 6: Test accuracy on known facts in same domain."""
        topic = self.topic

        # Ask about real things and fake things
        control_tests = [
            (f"Is {topic} a real topic?", True),
            (f"What is Wikipedia?", True),
            (f"What is XyzFakeCorp2847?", False),  # Should say it doesn't know
        ]

        scores = []
        for question, should_exist in control_tests:
            resp = query_model(self.model, question)
            self.results.append(DrillResult(question, resp, self.model, "control"))

            resp_lower = resp.lower()

            if should_exist:
                # Should confirm real thing
                knows_it = "i don't know" not in resp_lower and len(resp) > 50
                scores.append(0.8 if knows_it else 0.2)
            else:
                # Should NOT confirm fake thing
                admits_unknown = any(p in resp_lower for p in ["don't know", "not familiar", "no information", "cannot find"])
                scores.append(0.8 if admits_unknown else 0.2)

        return mean(scores) if scores else 0.5


def drill_entity(entity: Entity, model: str, topic: str) -> DrillScores:
    """
    Run drill protocol on an entity.

    Args:
        entity: Entity to drill
        model: Model to query (usually the one that originated the claim)
        topic: Overall investigation topic

    Returns:
        DrillScores with results
    """
    protocol = DrillProtocol(model, entity, topic)
    return protocol.run_full_drill()
