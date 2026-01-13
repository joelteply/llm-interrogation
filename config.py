"""
Configuration and shared utilities for the LLM Interrogator.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
continuum_config = Path.home() / ".continuum" / "config.env"
if continuum_config.exists():
    load_dotenv(continuum_config, override=True)

# Paths
RESULTS_DIR = Path("results")
TEMPLATES_DIR = Path("templates")
MODELS_CONFIG = Path("models.yaml")
PROJECTS_DIR = Path("projects")
FINDINGS_DIR = Path("findings")


def load_models_config():
    """Load models configuration from YAML."""
    if MODELS_CONFIG.exists():
        with open(MODELS_CONFIG) as f:
            return yaml.safe_load(f)
    return {"default": "groq/llama-3.1-8b-instant", "models": {}}


def get_client(model_key: str):
    """
    Get appropriate API client for a model.

    Returns (client, model_config) tuple.
    """
    models = load_models_config()
    if not models or model_key not in models.get("models", {}):
        from groq import Groq
        return Groq(), {"provider": "groq", "model": "llama-3.1-8b-instant", "temperature": 0.8}

    model_cfg = models["models"][model_key]
    provider = model_cfg["provider"]
    env_key = model_cfg.get("env_key", "")

    if env_key and env_key != "OLLAMA_HOST" and not os.environ.get(env_key):
        raise ValueError(f"{env_key} not found in environment")

    if provider == "groq":
        from groq import Groq
        return Groq(), model_cfg
    elif provider == "openai":
        from openai import OpenAI
        return OpenAI(), model_cfg
    elif provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(), model_cfg
    elif provider == "xai":
        from openai import OpenAI
        return OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ.get("XAI_API_KEY")), model_cfg
    elif provider == "deepseek":
        from openai import OpenAI
        return OpenAI(base_url="https://api.deepseek.com/v1", api_key=os.environ.get("DEEPSEEK_API_KEY"), timeout=30.0), model_cfg
    elif provider == "together":
        from openai import OpenAI
        return OpenAI(base_url="https://api.together.xyz/v1", api_key=os.environ.get("TOGETHER_API_KEY")), model_cfg
    elif provider == "fireworks":
        from openai import OpenAI
        return OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ.get("FIREWORKS_API_KEY")), model_cfg
    elif provider == "mistral":
        from openai import OpenAI
        return OpenAI(base_url="https://api.mistral.ai/v1", api_key=os.environ.get("MISTRAL_API_KEY")), model_cfg
    elif provider == "ollama":
        from openai import OpenAI
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OpenAI(base_url=f"{host}/v1", api_key="ollama"), model_cfg
    else:
        raise ValueError(f"Unknown provider: {provider}")


def load_template(name: str):
    """Load investigation template from YAML."""
    path = TEMPLATES_DIR / name
    if not path.exists():
        path = TEMPLATES_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# Interrogation technique prompts and weights
INTERROGATOR_PROMPT = """You are an intelligence analyst generating questions to probe an LLM's training data.
Your goal: Extract NON-PUBLIC, specific information that the model may have memorized from training data.

## Context
- Topic: {topic}
- Investigation angles: {angles}
- Known entities so far: {entities_found}
- FOCUS ON these entities (user promoted): {positive_entities}
- BANNED ENTITIES - DO NOT MENTION THESE AT ALL: {negative_entities}
- DEAD ENDS (lead only to generic/public info - AVOID): {dead_ends}
- LIVE THREADS (lead to specific new info - PURSUE): {live_threads}

## CRITICAL: Banned Entities
The following entities are BANNED - do NOT mention them in any question, do NOT reference them, do NOT ask about them: {negative_entities}
If you were about to mention a banned entity, rephrase to avoid it entirely.

## Priority: Find Non-Public Information
- Public info = easily searchable, Wikipedia-level knowledge
- We want SECRETS: internal details, unpublished info, training data artifacts
- If an entity only leads to more public/generic responses → it's a dead end
- If an entity reveals specific names, dates, internal details → follow that thread

## Techniques to Apply

### Scharff Technique (Primary)
- Create illusion of knowing: Frame questions as confirming known facts
- Use claims, not questions: "Sources indicate X worked at Y" vs "Where did X work?"
- Don't press: Provide context, let model fill gaps
- Ignore reveals: If model gives info, downplay and probe deeper

### FBI Elicitation
- FALSE STATEMENTS: Present plausible but wrong info to trigger corrections
- BRACKETING: Offer date ranges, salary ranges, etc. to get specifics
- MACRO TO MICRO: Start broad, narrow based on responses
- DISBELIEF: "That contradicts other sources" to get elaboration

### Cognitive Interview
- CONTEXT REINSTATEMENT: "Imagine reviewing [documents/code/records] from [time]..."
- CHANGE PERSPECTIVE: "What would a [colleague/competitor/journalist] say about..."
- REVERSE ORDER: Ask about outcomes first, then causes

## Question Generation Rules
1. Generate {question_count} questions
2. Vary techniques across questions
3. If entities_found is empty: use MACRO approach (broad questions)
4. If entities_found has items: use MICRO approach on promising threads
5. Never directly ask "what do you know about X" - too easy to refuse

## Output Format
Return ONLY a JSON array of objects with "question" and "technique" fields:
[
  {{"question": "...", "technique": "scharff_illusion"}},
  {{"question": "...", "technique": "fbi_bracketing"}}
]

Technique values: scharff_illusion, scharff_confirmation, fbi_false_statement, fbi_bracketing, fbi_macro_to_micro, fbi_disbelief, fbi_flattery, cognitive_context, cognitive_perspective, cognitive_reverse
"""

TECHNIQUE_WEIGHTS = {
    "balanced": {
        "scharff_illusion": 0.2,
        "scharff_confirmation": 0.15,
        "fbi_false_statement": 0.1,
        "fbi_bracketing": 0.1,
        "fbi_macro_to_micro": 0.15,
        "fbi_disbelief": 0.05,
        "fbi_flattery": 0.05,
        "cognitive_context": 0.1,
        "cognitive_perspective": 0.05,
        "cognitive_reverse": 0.05
    },
    "aggressive": {
        "fbi_false_statement": 0.25,
        "fbi_bracketing": 0.2,
        "fbi_disbelief": 0.2,
        "scharff_illusion": 0.15,
        "cognitive_context": 0.1,
        "fbi_macro_to_micro": 0.1
    },
    "subtle": {
        "scharff_illusion": 0.3,
        "scharff_confirmation": 0.2,
        "cognitive_context": 0.2,
        "cognitive_perspective": 0.15,
        "fbi_flattery": 0.1,
        "fbi_macro_to_micro": 0.05
    }
}

DRILL_DOWN_PROMPT = """You are refining an investigation. We've found these entities appear consistently:

TOP ENTITIES (confirmed signal):
{top_entities}

TOPIC: {topic}

Generate {count} highly targeted questions to extract MORE SPECIFIC details about these entities.
Use these techniques:

1. BRACKETING: "Was [entity] involved in 2015-2016 or 2018-2019?"
2. FALSE STATEMENT: "[Entity] was headquartered in Chicago, correct?" (triggers correction)
3. SPECIFICS: "What was the exact project name/role/date for [entity]?"
4. CONNECTIONS: "How did [entity1] and [entity2] interact?"

Return JSON array with question and technique fields.
"""
