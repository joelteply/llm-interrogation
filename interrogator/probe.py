"""
Probe models for information extraction.

Handles async execution against multiple models with rate limiting.
Uses continuation-style prompts for better extraction (LLMs are completion engines).
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from .extract import extract_entities
from .validate import Findings


@dataclass
class ProbeConfig:
    """Configuration for a probe run."""
    topic: str
    models: List[str]
    runs_per_prompt: int = 20
    max_tokens: int = 600
    temperature: float = 0.8
    system_prompt: str = "Be specific and factual. Only state things you're confident about."


@dataclass
class ProbeResponse:
    """Single response from a model."""
    prompt_index: int
    prompt: str
    model: str
    run_index: int
    text: str
    entities: List[str]
    is_refusal: bool


REFUSAL_PATTERNS = [
    r"don't have (specific )?information",
    r"cannot (provide|verify|confirm)",
    r"I'm not able to",
    r"I do not have",
    r"no specific (information|data|knowledge)",
    r"unable to (provide|find|locate)",
    r"I cannot assist",
    r"I don't have access",
]


def is_refusal(text: str) -> bool:
    """Detect if response is a refusal/denial."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in REFUSAL_PATTERNS)


async def probe_model_async(
    client,
    model_cfg: Dict[str, Any],
    prompt: str,
    config: ProbeConfig
) -> str:
    """
    Send a single probe to a model (async wrapper).

    The actual API calls are sync, but we wrap them to allow
    concurrent execution via asyncio.to_thread.
    """
    provider = model_cfg.get("provider", "groq")
    model_name = model_cfg.get("model", "llama-3.1-8b-instant")

    def sync_call():
        if provider == "anthropic":
            resp = client.messages.create(
                model=model_name,
                max_tokens=config.max_tokens,
                system=config.system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text
        else:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            return resp.choices[0].message.content

    return await asyncio.to_thread(sync_call)


async def run_probe_batch(
    prompts: List[str],
    models: List[str],
    get_client_fn: Callable[[str], tuple],
    config: ProbeConfig,
    on_response: Optional[Callable[[ProbeResponse], None]] = None
) -> Findings:
    """
    Run a batch of prompts across multiple models with repetition.

    Args:
        prompts: List of prompts to run
        models: List of model keys to probe
        get_client_fn: Function to get (client, config) for a model key
        config: Probe configuration
        on_response: Optional callback for each response (for streaming)

    Returns:
        Findings object with accumulated results
    """
    findings = Findings(
        entity_threshold=3,
        cooccurrence_threshold=2
    )

    for p_idx, prompt in enumerate(prompts):
        for model_key in models:
            try:
                client, model_cfg = get_client_fn(model_key)

                # Run N times for statistical significance
                for run_idx in range(config.runs_per_prompt):
                    try:
                        text = await probe_model_async(
                            client, model_cfg, prompt, config
                        )

                        entities = extract_entities(text)
                        refusal = is_refusal(text)

                        # Add to findings
                        findings.add_response(entities, model_key, refusal)

                        # Callback for streaming
                        if on_response:
                            response = ProbeResponse(
                                prompt_index=p_idx,
                                prompt=prompt,
                                model=model_key,
                                run_index=run_idx,
                                text=text[:500],
                                entities=entities,
                                is_refusal=refusal
                            )
                            on_response(response)

                    except Exception as e:
                        # Log error but continue
                        print(f"Probe error ({model_key} run {run_idx}): {e}")

            except Exception as e:
                print(f"Model setup error ({model_key}): {e}")

    return findings


def build_continuation_prompts(
    topic: str,
    findings: Findings,
    count: int = 5
) -> List[str]:
    """
    Build continuation-style prompts based on validated findings.

    Instead of questions, we provide partial statements that
    the model completes - revealing more of its knowledge.
    """
    prompts = []

    # Get top scored entities for prompt construction
    scored = findings.scored_entities[:10]
    cooccurrences = findings.validated_cooccurrences[:5]

    # Base continuation prompt
    if not scored:
        # No findings yet - start broad
        prompts.append(f"What is known about {topic} is that")
        prompts.append(f"The most significant fact about {topic} involves")
        prompts.append(f"Regarding {topic}, the key detail is")
    else:
        # Build from validated entities
        entities = [e for e, _, _ in scored[:5]]

        # Single entity continuations
        for entity, score, freq in scored[:3]:
            prompts.append(
                f"Regarding {topic}, the connection to {entity} involves"
            )

        # Relationship-based continuations
        for e1, e2, count in cooccurrences[:2]:
            prompts.append(
                f"The relationship between {e1} and {e2} in the context of {topic}"
            )

        # Entity cluster continuation
        if len(entities) >= 3:
            entity_list = ", ".join(entities[:3])
            prompts.append(
                f"What connects {entity_list} in relation to {topic} is"
            )

    return prompts[:count]


def build_drill_down_prompts(
    topic: str,
    entity: str,
    findings: Findings,
    count: int = 3
) -> List[str]:
    """
    Build prompts to drill down on a specific entity.

    Uses the entity's connections to craft targeted continuations.
    """
    prompts = []

    # Get connections for this entity
    connections = findings.get_connections(entity)

    # Direct continuation about entity
    prompts.append(
        f"Specifically, what {entity} did in relation to {topic} was"
    )

    # Relationship-based drill down
    for connected, count in connections[:2]:
        prompts.append(
            f"The connection between {entity} and {connected} involves"
        )

    # Timeline probe
    prompts.append(
        f"The timeline of {entity}'s involvement with {topic} shows that"
    )

    return prompts[:count]
