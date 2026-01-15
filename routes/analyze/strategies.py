"""
Strategy Loader - YAML-based configuration for document analysis.

Strategies define:
- What entities to look for
- How to frame questions (Mad Libs, trivia, etc.)
- Validation rules
- Reasoning guardrails
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

STRATEGIES_DIR = Path(__file__).parent.parent.parent / "strategies"


@dataclass
class Framing:
    """A question framing template."""
    name: str
    description: str
    prompt_template: str
    best_for_models: list[str] = field(default_factory=list)


@dataclass
class ValidationConfig:
    """Validation rules for responses."""
    require_specificity: bool = True
    require_multiple_sources: int = 2
    check_public_sources: bool = True
    max_confidence: int = 85


@dataclass
class Guardrails:
    """Reasoning guardrails to prevent confused AI output."""
    banned_concepts: list[str] = field(default_factory=list)
    banned_in_output: list[str] = field(default_factory=list)
    require_in_output: list[str] = field(default_factory=list)


@dataclass
class ResearchConfig:
    """Configuration for RAG/research phase."""
    enabled: bool = True
    search_terms: list[str] = field(default_factory=list)  # Keywords to search for
    search_domains: list[str] = field(default_factory=list)
    search_templates: list[str] = field(default_factory=list)
    entity_sources: list[str] = field(default_factory=list)
    bypass_techniques: list[str] = field(default_factory=list)


@dataclass
class Strategy:
    """Complete strategy configuration."""
    name: str
    description: str
    use_case: str  # unredaction, enrichment, discovery, etc.

    entity_types: list[str]
    techniques: list[str]
    framings: list[Framing]

    hints: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)

    validation: ValidationConfig = field(default_factory=ValidationConfig)
    guardrails: Guardrails = field(default_factory=Guardrails)
    research: ResearchConfig = field(default_factory=ResearchConfig)

    # Raw YAML for custom fields
    raw: dict = field(default_factory=dict)


def load_strategy(name: str) -> Strategy:
    """Load a strategy from YAML file."""
    # Handle both "name" and "name.yaml"
    if not name.endswith('.yaml'):
        name = f"{name}.yaml"

    path = STRATEGIES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Strategy not found: {name}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return _parse_strategy(data)


def _parse_strategy(data: dict) -> Strategy:
    """Parse raw YAML into Strategy dataclass."""

    # Parse framings
    framings = []
    for f in data.get('framings', []):
        if isinstance(f, dict):
            framings.append(Framing(
                name=f.get('name', 'default'),
                description=f.get('description', ''),
                prompt_template=f.get('prompt_template', ''),
                best_for_models=f.get('best_for_models', [])
            ))

    # Parse validation
    val_data = data.get('validation', {})
    validation = ValidationConfig(
        require_specificity=val_data.get('require_specificity', True),
        require_multiple_sources=val_data.get('require_multiple_sources', 2),
        check_public_sources=val_data.get('check_public_sources', True),
        max_confidence=val_data.get('max_confidence', 85)
    )

    # Parse guardrails
    guard_data = data.get('guardrails', data.get('reasoning_guardrails', {}))
    guardrails = Guardrails(
        banned_concepts=guard_data.get('banned_concepts', []),
        banned_in_output=guard_data.get('banned_in_output', []),
        require_in_output=guard_data.get('require_in_output', [])
    )

    # Parse research config
    research_data = data.get('research', {})
    research = ResearchConfig(
        enabled=research_data.get('enabled', True),
        search_terms=research_data.get('search_terms', []),
        search_domains=research_data.get('search_domains', []),
        search_templates=research_data.get('search_templates', []),
        entity_sources=research_data.get('entity_sources', []),
        bypass_techniques=research_data.get('bypass_techniques', [])
    )

    return Strategy(
        name=data.get('name', 'Unnamed Strategy'),
        description=data.get('description', ''),
        use_case=data.get('use_case', 'unredaction'),
        entity_types=data.get('entity_types', ['PERSON_NAME']),
        techniques=data.get('techniques', []),
        framings=framings,
        hints=data.get('hints', []),
        avoid=data.get('avoid', []),
        validation=validation,
        guardrails=guardrails,
        research=research,
        raw=data
    )


def list_strategies() -> list[dict]:
    """List all available strategies."""
    strategies = []

    if not STRATEGIES_DIR.exists():
        return strategies

    for path in STRATEGIES_DIR.glob("*.yaml"):
        if path.name.startswith('_'):  # Skip schema files
            continue
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            strategies.append({
                'id': path.stem,
                'name': data.get('name', path.stem),
                'description': data.get('description', ''),
                'use_case': data.get('use_case', 'unredaction')
            })
        except Exception as e:
            print(f"[STRATEGIES] Error loading {path}: {e}")

    return strategies


def save_strategy(name: str, data: dict) -> None:
    """Save a strategy to YAML file."""
    if not name.endswith('.yaml'):
        name = f"{name}.yaml"

    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    path = STRATEGIES_DIR / name

    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_strategy_raw(name: str) -> dict:
    """Get raw YAML content of a strategy."""
    if not name.endswith('.yaml'):
        name = f"{name}.yaml"

    path = STRATEGIES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Strategy not found: {name}")

    with open(path) as f:
        return yaml.safe_load(f)


def select_framing(strategy: Strategy, model: Optional[str] = None) -> Framing:
    """Select the best framing for a given model."""
    if not strategy.framings:
        # Return a default framing
        return Framing(
            name='default',
            description='Default question framing',
            prompt_template='Based on the context: "{context}", what {entity_type} would fit here?'
        )

    if model:
        # Look for model-specific framing
        for framing in strategy.framings:
            if any(m in model.lower() for m in framing.best_for_models):
                return framing

    # Return first framing as default
    return strategy.framings[0]
