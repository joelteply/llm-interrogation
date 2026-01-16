"""
Confidence Scoring

Calculates confidence scores for entities based on all available signals.
"""

from .entities import Entity


def calculate_confidence(entity: Entity) -> float:
    """
    Calculate confidence score for an entity.

    Factors:
    - Web corroboration
    - Independent model sources
    - Drill results
    - Echo chamber penalty
    - Leading question penalty
    """
    base = 1.0

    # Web corroboration bonus
    if entity.category == "CORROBORATED":
        base += 0.5
    elif entity.category == "CONTRADICTED":
        base *= 0.3  # Penalty for contradicted info

    # Multi-model confirmation bonus
    independent = entity.independent_sources
    if independent > 1:
        base += (0.2 * (independent - 1))  # +0.2 per additional independent source

    # Echo chamber penalty
    if entity.is_echo_chamber:
        base *= 0.3  # Heavy penalty for same model echoing itself

    # Single source slight penalty (but don't kill it - could be real leak)
    if entity.is_single_source and independent == 1:
        base *= 0.8

    # Drill results
    if entity.drill_scores:
        drill_avg = entity.drill_scores.average
        if drill_avg > 0.7:
            base *= 1.3  # Bonus for passing drill
        elif drill_avg > 0.5:
            base *= 1.0  # Neutral
        elif drill_avg > 0.3:
            base *= 0.7  # Penalty
        else:
            base *= 0.3  # Heavy penalty for failing drill

    # Normalize to 0-1 range (softmax-ish)
    confidence = base / (base + 1)

    return round(confidence, 3)


def confidence_label(confidence: float) -> str:
    """Get human-readable label for confidence level."""
    if confidence >= 0.7:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    elif confidence >= 0.3:
        return "LOW"
    else:
        return "VERY LOW"


def should_drill(entity: Entity) -> bool:
    """Determine if an entity should be drilled."""
    # Don't drill already drilled entities
    if entity.drill_scores is not None:
        return False

    # Drill uncorroborated entities
    if entity.category == "UNCORROBORATED":
        return True

    # Drill echo chambers even if "corroborated" by frequency
    if entity.is_echo_chamber:
        return True

    # Drill single-source claims with no web corroboration
    if entity.is_single_source and entity.category != "CORROBORATED":
        return True

    return False


def prioritize_for_drill(entities: list) -> list:
    """
    Prioritize entities for drilling.

    Higher priority:
    - More total mentions (potentially interesting)
    - Echo chambers (need verification)
    - Single source (need verification)
    """
    def priority_score(e: Entity) -> float:
        score = 0

        # More mentions = more interesting (but also more suspicious if single source)
        score += e.total_mentions * 0.1

        # Echo chambers are high priority to verify
        if e.is_echo_chamber:
            score += 5

        # Single source is priority
        if e.is_single_source:
            score += 3

        # Uncorroborated is priority
        if e.category == "UNCORROBORATED":
            score += 2

        return score

    drillable = [e for e in entities if should_drill(e)]
    return sorted(drillable, key=priority_score, reverse=True)
