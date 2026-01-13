"""
AI-driven narrative synthesis from validated findings.

Takes validated entities and co-occurrences, asks AI to synthesize
a coherent narrative about the topic.
"""

from typing import List, Tuple, Optional
from .validate import Findings


def build_synthesis_prompt(
    topic: str,
    findings: Findings,
    max_entities: int = 20,
    max_cooccurrences: int = 15
) -> str:
    """
    Build a prompt for the AI to synthesize findings into narrative.

    Uses scored entities (complex concepts ranked higher) and
    validated co-occurrences to give the AI structured input.
    """
    # Get top scored entities
    scored = findings.scored_entities[:max_entities]

    # Get validated relationships
    cooccurrences = findings.validated_cooccurrences[:max_cooccurrences]

    # Format entities with scores
    entities_str = "\n".join([
        f"- {entity} (mentioned {freq}x, score: {score:.1f})"
        for entity, score, freq in scored
    ])

    # Format relationships
    relationships_str = "\n".join([
        f"- {e1} <-> {e2} (appeared together {count}x)"
        for e1, e2, count in cooccurrences
    ])

    prompt = f"""Based on probing language models about "{topic}", the following concepts and relationships emerged consistently (validated through repetition across {findings.corpus_size} responses):

## Key Concepts (ranked by importance)
{entities_str}

## Validated Relationships (concepts that appeared together)
{relationships_str}

## Task
Synthesize these findings into a coherent narrative about {topic}.
- Focus on the high-scoring concepts (they are more specific and meaningful)
- Use the relationships to connect concepts logically
- Write 2-3 paragraphs that tell the story these data points suggest
- Only include information supported by the validated data above
- If relationships suggest a timeline or progression, capture that

Write the narrative:"""

    return prompt


def build_continuation_prompt(
    topic: str,
    findings: Findings,
    focus_entities: Optional[List[str]] = None
) -> str:
    """
    Build a continuation prompt using validated entities.

    Instead of asking questions, we provide a partial statement
    and let the model complete it - revealing more of its knowledge.
    """
    if focus_entities:
        # Focus on specific cluster
        entities = focus_entities
    else:
        # Use top scored entities
        entities = [e for e, _, _ in findings.scored_entities[:5]]

    if not entities:
        return f"What is known about {topic} is that"

    # Build a partial sentence from the validated entities
    if len(entities) == 1:
        return f"Regarding {topic}, the connection to {entities[0]} involves"
    elif len(entities) == 2:
        return f"The relationship between {entities[0]} and {entities[1]} in the context of {topic}"
    else:
        entity_list = ", ".join(entities[:-1]) + f" and {entities[-1]}"
        return f"What connects {entity_list} in relation to {topic} is"


def build_cluster_probe_prompt(
    topic: str,
    cluster_entities: List[str],
    cluster_relationships: List[Tuple[str, str, int]]
) -> str:
    """
    Build a prompt to probe deeper into a specific cluster.

    Uses the cluster's entities and relationships to craft
    a focused continuation prompt.
    """
    entities_str = ", ".join(cluster_entities[:5])

    # Find the strongest relationship in cluster
    if cluster_relationships:
        e1, e2, count = cluster_relationships[0]
        relationship_hint = f" (particularly the connection between {e1} and {e2})"
    else:
        relationship_hint = ""

    return f"""In researching {topic}, consistent evidence points to involvement with {entities_str}{relationship_hint}.

Specifically, this suggests"""


def extract_narrative_entities(narrative: str) -> List[str]:
    """
    Extract any new entities mentioned in a synthesized narrative.

    These can be added to the investigation for the next cycle.
    """
    from .extract import extract_entities
    return extract_entities(narrative)
