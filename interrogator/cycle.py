"""
Extraction cycle orchestration.

The core loop: PROBE → VALIDATE → CONDENSE → GROW → repeat

Each cycle:
1. PROBE: Run prompts against models N times
2. VALIDATE: Statistical validation separates signal from noise
3. CONDENSE: AI synthesizes findings into coherent narrative
4. GROW: Use narrative to generate new continuation prompts
5. Repeat with enriched context
"""

from typing import List, Dict, Any, Optional, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum

from .validate import Findings
from .synthesize import (
    build_synthesis_prompt,
    build_continuation_prompt,
    build_cluster_probe_prompt,
    extract_narrative_entities
)
from .probe import (
    ProbeConfig,
    ProbeResponse,
    build_continuation_prompts,
    build_drill_down_prompts
)


class CyclePhase(Enum):
    """Phases of the extraction cycle."""
    PROBE = "probe"
    VALIDATE = "validate"
    CONDENSE = "condense"
    GROW = "grow"


@dataclass
class CycleState:
    """
    State of an extraction cycle.

    Accumulates findings across cycles and tracks what's been explored.
    """
    topic: str
    findings: Findings = field(default_factory=Findings)
    narratives: List[str] = field(default_factory=list)
    explored_entities: set = field(default_factory=set)
    cycle_count: int = 0
    dead_ends: set = field(default_factory=set)  # Entities that led nowhere

    def should_continue(self, max_cycles: int = 5) -> bool:
        """
        Determine if cycle should continue.

        Stop if:
        - Max cycles reached
        - No new validated entities in last cycle
        - High refusal rate indicates dead end
        """
        if self.cycle_count >= max_cycles:
            return False

        # Check refusal rate
        if self.findings.refusal_rate > 0.8:
            return False

        # Check if we're still finding new entities
        if self.cycle_count > 1 and len(self.findings.validated_entities) == 0:
            return False

        return True

    def get_unexplored_entities(self, limit: int = 5) -> List[str]:
        """Get high-scoring entities that haven't been explored yet."""
        unexplored = []
        for entity, score, freq in self.findings.scored_entities:
            if entity not in self.explored_entities and entity not in self.dead_ends:
                unexplored.append(entity)
                if len(unexplored) >= limit:
                    break
        return unexplored

    def mark_explored(self, entity: str):
        """Mark an entity as explored."""
        self.explored_entities.add(entity)

    def mark_dead_end(self, entity: str):
        """Mark an entity as a dead end (high refusal rate)."""
        self.dead_ends.add(entity)


@dataclass
class CycleEvent:
    """Event emitted during cycle execution."""
    phase: CyclePhase
    event_type: str  # e.g., "start", "response", "complete"
    data: Any


def run_extraction_cycle(
    state: CycleState,
    models: List[str],
    get_client_fn: Callable[[str], tuple],
    synthesize_fn: Callable[[str], str],
    config: Optional[ProbeConfig] = None
) -> Generator[CycleEvent, None, CycleState]:
    """
    Run a single extraction cycle.

    Yields events for streaming progress.
    Returns updated state.

    Args:
        state: Current cycle state
        models: Models to probe
        get_client_fn: Function to get (client, config) for model key
        synthesize_fn: Function to call AI for synthesis (takes prompt, returns response)
        config: Probe configuration

    Yields:
        CycleEvent objects for each phase

    Returns:
        Updated CycleState
    """
    if config is None:
        config = ProbeConfig(
            topic=state.topic,
            models=models,
            runs_per_prompt=20
        )

    state.cycle_count += 1

    # ==================== PHASE 1: PROBE ====================
    yield CycleEvent(
        phase=CyclePhase.PROBE,
        event_type="start",
        data={"cycle": state.cycle_count}
    )

    # Build prompts based on current state
    if state.cycle_count == 1:
        # First cycle: broad continuation prompts
        prompts = build_initial_prompts(state.topic)
    else:
        # Subsequent cycles: use findings to build targeted prompts
        prompts = build_continuation_prompts(
            state.topic,
            state.findings,
            count=5
        )

        # Add drill-down prompts for unexplored entities
        for entity in state.get_unexplored_entities(limit=2):
            prompts.extend(
                build_drill_down_prompts(state.topic, entity, state.findings, count=2)
            )
            state.mark_explored(entity)

    yield CycleEvent(
        phase=CyclePhase.PROBE,
        event_type="prompts",
        data={"prompts": prompts}
    )

    # Note: Actual probing happens in the caller's async context
    # This generator yields the prompts, caller runs them and feeds back findings

    yield CycleEvent(
        phase=CyclePhase.PROBE,
        event_type="ready",
        data={"prompts": prompts, "config": config}
    )

    # ==================== PHASE 2: VALIDATE ====================
    # Validation happens automatically via Findings.add_response()
    # The caller will have added responses to state.findings

    yield CycleEvent(
        phase=CyclePhase.VALIDATE,
        event_type="start",
        data={}
    )

    validated = state.findings.validated_entities
    noise = state.findings.noise_entities
    cooccurrences = state.findings.validated_cooccurrences

    yield CycleEvent(
        phase=CyclePhase.VALIDATE,
        event_type="complete",
        data={
            "validated_count": len(validated),
            "noise_count": len(noise),
            "cooccurrence_count": len(cooccurrences),
            "refusal_rate": state.findings.refusal_rate
        }
    )

    # ==================== PHASE 3: CONDENSE ====================
    if len(validated) > 0:
        yield CycleEvent(
            phase=CyclePhase.CONDENSE,
            event_type="start",
            data={}
        )

        # Build synthesis prompt
        synthesis_prompt = build_synthesis_prompt(
            state.topic,
            state.findings,
            max_entities=20,
            max_cooccurrences=15
        )

        yield CycleEvent(
            phase=CyclePhase.CONDENSE,
            event_type="synthesizing",
            data={"prompt_length": len(synthesis_prompt)}
        )

        # Call AI for synthesis (the caller provides this function)
        try:
            narrative = synthesize_fn(synthesis_prompt)
            state.narratives.append(narrative)

            yield CycleEvent(
                phase=CyclePhase.CONDENSE,
                event_type="complete",
                data={"narrative": narrative}
            )
        except Exception as e:
            yield CycleEvent(
                phase=CyclePhase.CONDENSE,
                event_type="error",
                data={"error": str(e)}
            )

    # ==================== PHASE 4: GROW ====================
    yield CycleEvent(
        phase=CyclePhase.GROW,
        event_type="start",
        data={}
    )

    # Extract new entities from narrative (if we have one)
    new_entities = []
    if state.narratives:
        new_entities = extract_narrative_entities(state.narratives[-1])

        # Add new entities to findings for next cycle
        # These get validation score of 1 (unvalidated, from synthesis)
        for entity in new_entities:
            if entity not in state.findings.entity_counts:
                state.findings.entity_counts[entity] = 1

    # Determine if we should continue
    should_continue = state.should_continue()

    yield CycleEvent(
        phase=CyclePhase.GROW,
        event_type="complete",
        data={
            "new_entities": new_entities,
            "should_continue": should_continue,
            "next_targets": state.get_unexplored_entities(limit=3)
        }
    )

    return state


def build_initial_prompts(topic: str, count: int = 5) -> List[str]:
    """
    Build initial broad prompts for first cycle.

    Uses continuation style rather than questions.
    """
    return [
        f"What is known about {topic} is that",
        f"The most significant fact about {topic} involves",
        f"Regarding {topic}, the key detail is",
        f"The context around {topic} includes",
        f"Information about {topic} reveals that",
    ][:count]


def cluster_entities(findings: Findings) -> List[List[str]]:
    """
    Cluster entities based on co-occurrence patterns.

    Entities that frequently appear together likely belong to the same concept/thread.
    """
    # Build adjacency from cooccurrences
    adjacency: Dict[str, set] = {}
    for e1, e2, count in findings.validated_cooccurrences:
        if count >= 2:  # Only strong connections
            adjacency.setdefault(e1, set()).add(e2)
            adjacency.setdefault(e2, set()).add(e1)

    # Simple connected components (clustering)
    visited = set()
    clusters = []

    for entity in findings.validated_entities:
        if entity in visited:
            continue

        # BFS to find connected component
        cluster = []
        queue = [entity]
        while queue:
            e = queue.pop(0)
            if e in visited:
                continue
            visited.add(e)
            cluster.append(e)

            for neighbor in adjacency.get(e, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        if cluster:
            clusters.append(cluster)

    # Sort clusters by total frequency (most significant first)
    def cluster_score(c):
        return sum(findings.entity_counts.get(e, 0) for e in c)

    clusters.sort(key=cluster_score, reverse=True)
    return clusters


def identify_threads(
    findings: Findings,
    min_cluster_size: int = 2
) -> List[Dict[str, Any]]:
    """
    Identify narrative threads from clustered entities.

    A thread is a group of related entities that form a coherent sub-topic.
    """
    clusters = cluster_entities(findings)
    threads = []

    for i, cluster in enumerate(clusters):
        if len(cluster) < min_cluster_size:
            continue

        # Get relationships within cluster
        cluster_relationships = [
            (e1, e2, c)
            for e1, e2, c in findings.validated_cooccurrences
            if e1 in cluster and e2 in cluster
        ]

        # Score the thread
        total_freq = sum(findings.entity_counts.get(e, 0) for e in cluster)
        total_connections = len(cluster_relationships)

        threads.append({
            "id": i,
            "entities": cluster,
            "relationships": cluster_relationships,
            "score": total_freq * (1 + total_connections * 0.2),
            "size": len(cluster)
        })

    # Sort by score
    threads.sort(key=lambda t: t["score"], reverse=True)
    return threads
