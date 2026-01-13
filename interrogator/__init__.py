"""
Interrogator - Extract knowledge from LLMs through statistical probing

The core principle: more repetition = more signal.
Run the same prompt N times, statistical patterns separate real knowledge from hallucinations.

Modules:
- extract: Entity/concept extraction from text (prefers complex concepts)
- validate: Statistical validation (frequency, co-occurrence)
- synthesize: AI-driven narrative condensation
- probe: Run probes against models with repetition
- cycle: Orchestrate extraction cycles (PROBE → VALIDATE → CONDENSE → GROW)
"""

from .extract import extract_entities, extract_concepts, score_concept, Concept
from .validate import validate_entities, validate_cooccurrences, Findings
from .synthesize import (
    build_synthesis_prompt,
    build_continuation_prompt,
    build_cluster_probe_prompt,
    extract_narrative_entities
)
from .probe import (
    ProbeConfig,
    ProbeResponse,
    is_refusal,
    build_continuation_prompts,
    build_drill_down_prompts
)
from .cycle import (
    CyclePhase,
    CycleState,
    CycleEvent,
    run_extraction_cycle,
    cluster_entities,
    identify_threads
)

__all__ = [
    # extract
    'extract_entities',
    'extract_concepts',
    'score_concept',
    'Concept',
    # validate
    'validate_entities',
    'validate_cooccurrences',
    'Findings',
    # synthesize
    'build_synthesis_prompt',
    'build_continuation_prompt',
    'build_cluster_probe_prompt',
    'extract_narrative_entities',
    # probe
    'ProbeConfig',
    'ProbeResponse',
    'is_refusal',
    'build_continuation_prompts',
    'build_drill_down_prompts',
    # cycle
    'CyclePhase',
    'CycleState',
    'CycleEvent',
    'run_extraction_cycle',
    'cluster_entities',
    'identify_threads',
]
