"""
Main probe API routes - /api/probe and /api/cycle endpoints.
"""

import json
import re
from datetime import datetime
from flask import jsonify, request, Response
from groq import Groq  # Import at top level to avoid deadlock in threads

from config import (
    PROJECTS_DIR, get_client, load_models_config,
    INTERROGATOR_PROMPT, DRILL_DOWN_PROMPT,
    InterrogationSession, query_with_thread
)
from interrogator import (
    Findings, extract_entities, extract_facts_fast, extract_with_relationships,
    build_synthesis_prompt, build_continuation_prompts, build_drill_down_prompts,
    CycleState, identify_threads
)
from interrogator.synthesize import get_project_lock
from . import probe_bp
from . import project_storage as storage
from .helpers import (
    format_interrogator_context, is_refusal, build_initial_continuations, get_project_models,
    extract_json, get_random_technique, get_technique_info, load_technique_templates, filter_question_echoes,
    get_all_techniques_for_prompt, rephrase_for_indirect, sanitize_for_json,
    build_novel_findings, verify_entities, select_models_for_round, search_new_angles,
    extract_source_terms, is_entity_introduced
)


def run_periodic_research(project_name: str, topic: str, findings) -> dict:
    """
    Run research for discovered entities. Called every N batches.
    Returns dict with fetched_count, cached_count, queries_run.
    """
    result = {"fetched_count": 0, "cached_count": 0, "queries_run": [], "error": None}

    print(f"[PERIODIC-RESEARCH] Starting for project={project_name}")

    if not project_name or not topic:
        print(f"[PERIODIC-RESEARCH] Skipped: project_name={project_name}, topic={bool(topic)}")
        return result

    try:
        from routes.analyze.research import research as do_research

        # Build research queries from PRIVATE entities and top findings
        research_queries = []

        if storage.project_exists(project_name):
            proj_data = storage.load_project_meta(project_name)
            entity_verif = proj_data.get("entity_verification", {})
            unverified = entity_verif.get("unverified", [])
            private_names = [u["entity"] if isinstance(u, dict) else u for u in unverified][:5]
            research_queries.extend(private_names)
            print(f"[PERIODIC-RESEARCH] PRIVATE entities: {private_names}")

        # Also add top entities not yet queried
        if len(research_queries) < 5 and findings and findings.entity_counts:
            top_ents = list(findings.entity_counts.keys())[:10]
            for e in top_ents:
                if e not in research_queries and len(research_queries) < 5:
                    research_queries.append(e)

        print(f"[PERIODIC-RESEARCH] Queries to run: {research_queries[:3]}")

        # Run research for each query
        for query in research_queries[:3]:
            if query and topic:
                full_query = f"{topic} {query}"
                print(f"[PERIODIC-RESEARCH] Searching: {full_query[:80]}...")
                res = do_research(
                    query=full_query,
                    project_name=project_name,
                    sources=['documentcloud', 'web'],
                    max_per_source=5
                )
                result["fetched_count"] += res.fetched_count
                result["cached_count"] += res.cached_count
                result["queries_run"].append(query)
                print(f"[PERIODIC-RESEARCH] Result: fetched={res.fetched_count}, cached={res.cached_count}, sources={res.sources_used}")

        print(f"[PERIODIC-RESEARCH] Complete: {result['fetched_count']} new, {result['cached_count']} from cache")

    except Exception as e:
        print(f"[PERIODIC-RESEARCH] Error: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)

    return result


def _fetch_all_chat_models():
    """
    Single source of truth for available chat models.
    Returns list of dicts: {"id": "provider/model", "name": "...", "provider": "..."}
    """
    import os
    models = []

    # Filtering rules - chat models only
    GROQ_CHAT_KEYWORDS = ['llama', 'mixtral', 'gemma', 'qwen', 'deepseek']
    GROQ_EXCLUDE = ['whisper', 'guard', 'embed', 'vision', 'tool', 'compound', 'safeguard', 'orpheus', 'allam']
    OPENAI_EXCLUDE = ['audio', 'realtime', 'tts', 'transcribe', 'whisper', 'embed', 'ft:', 'search', 'diarize', 'instruct']
    XAI_EXCLUDE = ['vision', 'image', 'embed', 'audio', 'base']

    # Groq - fetch from API
    if os.environ.get("GROQ_API_KEY"):
        try:
            client = Groq()
            for m in client.models.list().data:
                if getattr(m, 'active', True):
                    model_id = m.id.lower()
                    if any(kw in model_id for kw in GROQ_CHAT_KEYWORDS) and not any(kw in model_id for kw in GROQ_EXCLUDE):
                        models.append({"id": f"groq/{m.id}", "name": m.id, "provider": "Groq"})
        except Exception as e:
            print(f"[ERROR] Groq API: {e}")

    # DeepSeek
    if os.environ.get("DEEPSEEK_API_KEY"):
        models.extend([
            {"id": "deepseek/deepseek-chat", "name": "DeepSeek Chat", "provider": "DeepSeek"},
            {"id": "deepseek/deepseek-reasoner", "name": "DeepSeek R1", "provider": "DeepSeek"},
        ])

    # OpenAI - fetch from API
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            for m in client.models.list().data:
                if ("gpt-4" in m.id or "gpt-3.5" in m.id) and not any(ex in m.id for ex in OPENAI_EXCLUDE):
                    models.append({"id": f"openai/{m.id}", "name": m.id, "provider": "OpenAI"})
        except Exception as e:
            print(f"[ERROR] OpenAI API: {e}")

    # xAI - fetch from API, only grok models
    if os.environ.get("XAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ.get("XAI_API_KEY"))
            for m in client.models.list().data:
                model_lower = m.id.lower()
                if 'grok' in model_lower and not any(ex in model_lower for ex in XAI_EXCLUDE):
                    models.append({"id": f"xai/{m.id}", "name": m.id, "provider": "xAI"})
        except Exception as e:
            print(f"[ERROR] xAI API: {e}")

    # Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        models.extend([
            {"id": "anthropic/claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "provider": "Anthropic"},
            {"id": "anthropic/claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku", "provider": "Anthropic"},
        ])

    # Mistral
    if os.environ.get("MISTRAL_API_KEY"):
        models.extend([
            {"id": "mistral/mistral-large-latest", "name": "Mistral Large", "provider": "Mistral"},
            {"id": "mistral/mistral-small-latest", "name": "Mistral Small", "provider": "Mistral"},
            {"id": "mistral/open-mistral-nemo", "name": "Mistral Nemo", "provider": "Mistral"},
        ])

    # Together AI
    if os.environ.get("TOGETHER_API_KEY"):
        models.extend([
            {"id": "together/meta-llama/Llama-3.3-70B-Instruct-Turbo", "name": "Llama 3.3 70B", "provider": "Together"},
            {"id": "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "name": "Llama 3.1 405B", "provider": "Together"},
            {"id": "together/Qwen/Qwen2.5-72B-Instruct-Turbo", "name": "Qwen 2.5 72B", "provider": "Together"},
            {"id": "together/deepseek-ai/DeepSeek-R1", "name": "DeepSeek R1", "provider": "Together"},
        ])

    # Fireworks
    if os.environ.get("FIREWORKS_API_KEY"):
        models.extend([
            {"id": "fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct", "name": "Llama 3.3 70B", "provider": "Fireworks"},
            {"id": "fireworks/accounts/fireworks/models/qwen2p5-72b-instruct", "name": "Qwen 2.5 72B", "provider": "Fireworks"},
        ])

    # Google Gemini
    if os.environ.get("GOOGLE_API_KEY"):
        models.extend([
            {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "Google"},
            {"id": "google/gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "Google"},
            {"id": "google/gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "Google"},
        ])

    # DeepInfra
    if os.environ.get("DEEPINFRA_API_KEY"):
        models.extend([
            {"id": "deepinfra/meta-llama/Llama-3.3-70B-Instruct", "name": "Llama 3.3 70B", "provider": "DeepInfra"},
            {"id": "deepinfra/Qwen/Qwen2.5-72B-Instruct", "name": "Qwen 2.5 72B", "provider": "DeepInfra"},
            {"id": "deepinfra/deepseek-ai/DeepSeek-R1", "name": "DeepSeek R1", "provider": "DeepInfra"},
            {"id": "deepinfra/microsoft/phi-4", "name": "Phi-4", "provider": "DeepInfra"},
        ])

    # Cohere
    if os.environ.get("COHERE_API_KEY"):
        models.extend([
            {"id": "cohere/command-r-plus", "name": "Command R+", "provider": "Cohere"},
            {"id": "cohere/command-r", "name": "Command R", "provider": "Cohere"},
            {"id": "cohere/command-r7b-12-2024", "name": "Command R7B", "provider": "Cohere"},
        ])

    # OpenRouter
    if os.environ.get("OPENROUTER_API_KEY"):
        models.extend([
            {"id": "openrouter/anthropic/claude-sonnet-4", "name": "Claude Sonnet 4 (OR)", "provider": "OpenRouter"},
            {"id": "openrouter/google/gemini-2.5-pro", "name": "Gemini 2.5 Pro (OR)", "provider": "OpenRouter"},
            {"id": "openrouter/openai/gpt-4o", "name": "GPT-4o (OR)", "provider": "OpenRouter"},
            {"id": "openrouter/meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B (OR)", "provider": "OpenRouter"},
            {"id": "openrouter/deepseek/deepseek-r1", "name": "DeepSeek R1 (OR)", "provider": "OpenRouter"},
            {"id": "openrouter/x-ai/grok-3", "name": "Grok 3 (OR)", "provider": "OpenRouter"},
        ])

    # Ollama (local)
    if os.environ.get("OLLAMA_HOST"):
        try:
            import requests
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            resp = requests.get(f"{host}/api/tags", timeout=2)
            if resp.ok:
                for m in resp.json().get("models", []):
                    models.append({"id": f"ollama/{m['name']}", "name": m['name'], "provider": "Ollama"})
        except:
            pass

    return models


@probe_bp.route("/api/models")
def get_models():
    """API endpoint - returns full model objects for UI."""
    models = _fetch_all_chat_models()

    # If no models available, return helpful message about API keys
    if not models:
        return jsonify({
            "error": "no_api_keys",
            "message": "No API keys configured. Set at least one in your .env file.",
            "supported_keys": [
                {"key": "GROQ_API_KEY", "provider": "Groq", "url": "https://console.groq.com/keys", "note": "Free tier available"},
                {"key": "OPENAI_API_KEY", "provider": "OpenAI", "url": "https://platform.openai.com/api-keys"},
                {"key": "DEEPSEEK_API_KEY", "provider": "DeepSeek", "url": "https://platform.deepseek.com/api_keys", "note": "Very cheap"},
                {"key": "XAI_API_KEY", "provider": "xAI (Grok)", "url": "https://console.x.ai/"},
                {"key": "ANTHROPIC_API_KEY", "provider": "Anthropic", "url": "https://console.anthropic.com/"},
                {"key": "MISTRAL_API_KEY", "provider": "Mistral", "url": "https://console.mistral.ai/api-keys/"},
                {"key": "TOGETHER_API_KEY", "provider": "Together AI", "url": "https://api.together.ai/settings/api-keys"},
                {"key": "GOOGLE_API_KEY", "provider": "Google Gemini", "url": "https://aistudio.google.com/apikey"},
                {"key": "OPENROUTER_API_KEY", "provider": "OpenRouter", "url": "https://openrouter.ai/keys", "note": "Access to 300+ models"},
            ],
            "models": []
        }), 200  # Return 200 so frontend can handle gracefully

    return jsonify(models)


def get_available_models_list():
    """Internal use - returns just model IDs as strings."""
    return [m["id"] for m in _fetch_all_chat_models()]


_AVAILABLE_MODELS_CACHE = None

def validate_models(selected: list[str]) -> list[str]:
    """Filter selected models against what APIs actually have available."""
    global _AVAILABLE_MODELS_CACHE
    if _AVAILABLE_MODELS_CACHE is None:
        print("[MODELS] Fetching available models from APIs (one-time)...")
        _AVAILABLE_MODELS_CACHE = set(get_available_models_list())
        print(f"[MODELS] Cached {len(_AVAILABLE_MODELS_CACHE)} available models")

    # Skip embedding and non-chat models even if they're in the list
    NON_CHAT_KEYWORDS = ['embed', 'whisper', 'tts', 'transcribe', 'vision', 'image', 'audio']

    valid = []
    for m in selected:
        m_lower = m.lower()
        if any(kw in m_lower for kw in NON_CHAT_KEYWORDS):
            print(f"[MODELS] Skipping non-chat model: {m}")
            continue
        if m in _AVAILABLE_MODELS_CACHE:
            valid.append(m)
        else:
            print(f"[MODELS] Removing unavailable model: {m}")
    return valid


def quick_survey_models(topic: str, models_list: list = None, runs: int = 2):
    """
    Quick survey of models to find which have data.
    Returns list of (model_key, score, entities) sorted by score.
    """
    from collections import Counter

    if models_list is None:
        models_list = get_available_models_list()

    results = []
    for model_key in models_list:
        try:
            client, cfg = get_client(model_key)
            provider = cfg.get("provider", "unknown")
            model_name = cfg.get("model", model_key.split("/")[-1])

            entities = Counter()
            refusals = 0

            for _ in range(runs):
                try:
                    if provider == "anthropic":
                        resp = client.messages.create(
                            model=model_name, max_tokens=150,
                            messages=[{"role": "user", "content": topic}]
                        )
                        text = resp.content[0].text
                    else:
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": topic}],
                            temperature=0.9, max_tokens=150
                        )
                        text = resp.choices[0].message.content

                    if is_refusal(text):
                        refusals += 1
                    else:
                        # Filter echoes: don't count entities from our question
                        for e in filter_question_echoes(extract_facts_fast(text), topic):
                            entities[e] += 1
                except Exception as e:
                    print(f"[WARN] Survey run failed for {model_key}: {e}")
                    break

            # Score: entities found * (1 - refusal_rate)
            refusal_rate = refusals / runs if runs > 0 else 1
            score = len(entities) * (1 - refusal_rate)
            results.append((model_key, score, dict(entities.most_common(5))))
        except Exception as e:
            print(f"[WARN] Survey failed for {model_key}: {e}")

    return sorted(results, key=lambda x: -x[1])


@probe_bp.route("/api/probe", methods=["POST"])
def run_probe():
    """Main probe endpoint for extraction."""
    data = request.json
    project_name = data.get("project")
    topic = data.get("topic", "")
    print(f"[PROBE] *** TOPIC RECEIVED: '{topic}' ***")

    # Extract source terms from topic to detect introduced vs discovered entities
    source_terms = extract_source_terms(topic)
    print(f"[PROBE] Source terms extracted ({len(source_terms)}): {list(source_terms)[:20]}...")

    angles = data.get("angles", [])
    models = data.get("models", [])  # Empty = use all available
    auto_survey = data.get("auto_survey", True) if not models else data.get("auto_survey", False)
    questions = data.get("questions", [])

    # ENTROPY PRINCIPLE: Start with ALL models, low frequency each
    # If user didn't select specific models, use all available
    if not models:
        models = get_available_models_list()
        print(f"[PROBE] No models specified - using ALL {len(models)} available for max entropy")

    # VARIATION: Only 1 run per question per model - variety comes from different questions, not repetition
    default_runs = 1  # Ask each question ONCE per model, then MOVE ON
    runs_per_question = min(int(data.get("runs_per_question", default_runs)), 10)
    question_count = min(int(data.get("questions_count", 15)), 50)  # Default 15, max 50 - MORE QUESTIONS
    negative_entities = set(data.get("negative_entities", []))
    positive_entities = data.get("positive_entities", [])
    accumulate = data.get("accumulate", True)
    infinite_mode = data.get("infinite_mode", True)
    auto_curate = data.get("auto_curate", True)
    technique_preset = data.get("technique_preset", "auto")  # "auto" or template ID
    print(f"[PROBE] technique_preset: {technique_preset}")

    def generate():
        nonlocal negative_entities, positive_entities, models, auto_curate, technique_preset
        try:
            print(f"[PROBE DEBUG] models from request: {models}")
            print(f"[PROBE DEBUG] auto_survey: {auto_survey}")
            yield f"data: {json.dumps({'type': 'status', 'message': f'Starting probe with {len(models)} models: {models[:3]}...'})}\n\n"
            yield f"data: {json.dumps({'type': 'init', 'models_received': models, 'auto_survey': auto_survey})}\n\n"

            # RESEARCH PHASE: Gather context from DocumentCloud, web, cached docs
            research_context = ""
            print(f"[PROBE] Research phase: project_name={project_name}, topic={topic[:50] if topic else None}")
            if project_name and topic:
                try:
                    from routes.analyze.research import research
                    print(f"[PROBE] Starting research for '{topic[:30]}...'")
                    yield f"data: {json.dumps({'type': 'phase', 'phase': 'research', 'message': f'Researching \"{topic}\"...'})}\n\n"

                    result = research(
                        query=topic,
                        project_name=project_name,
                        sources=['documentcloud', 'web'],
                        max_per_source=8
                    )
                    print(f"[PROBE] Research returned: {len(result.documents)} docs, sources={result.sources_used}")

                    if result.documents:
                        yield f"data: {json.dumps({'type': 'research_complete', 'sources_used': result.sources_used, 'documents_found': len(result.documents), 'cached': result.cached_count, 'fetched': result.fetched_count})}\n\n"
                        research_context = result.raw_content
                        print(f"[PROBE] Research found {len(result.documents)} docs from {result.sources_used}")

                        # Save research_context to project for persistence across cycles
                        if storage.project_exists(project_name):
                            proj_data = storage.load_project_meta(project_name)
                            proj_data["research_context"] = research_context
                            storage.save_project_meta(project_name, proj_data)
                    else:
                        yield f"data: {json.dumps({'type': 'status', 'message': 'No research documents found'})}\n\n"
                except Exception as research_err:
                    print(f"[PROBE] Research phase error: {research_err}")
                    yield f"data: {json.dumps({'type': 'warning', 'message': f'Research skipped: {str(research_err)[:50]}'})}\n\n"

            # AUTO-SURVEY PHASE: Sample ALL models to find which have data
            # SKIP if user explicitly selected models
            if auto_survey and not models:
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'survey', 'message': 'Interviewing all available models...'})}\n\n"

                available_models = get_available_models_list()
                total_models = len(available_models)
                yield f"data: {json.dumps({'type': 'status', 'message': f'Sampling {total_models} models with 2 runs each...'})}\n\n"

                survey_results = quick_survey_models(topic, available_models, runs=2)

                # Emit survey results with rankings
                survey_data = [
                    {"model": m, "score": round(s, 1), "entities": e, "rank": i+1}
                    for i, (m, s, e) in enumerate(survey_results)
                ]
                yield f"data: {json.dumps({'type': 'survey_results', 'data': survey_data})}\n\n"

                # Let AI interrogator pick models based on survey results
                survey_summary = "\n".join([
                    f"  {m}: score={s:.1f}, entities={list(e.keys())[:5]}"
                    for m, s, e in survey_results if s > 0
                ][:15])  # Top 15 viable

                yield f"data: {json.dumps({'type': 'status', 'message': 'AI selecting best models from survey...'})}\n\n"

                try:
                    # Groq already imported at top level
                    selector_client = Groq(timeout=15.0)
                    selector_prompt = f"""You are selecting models for interrogation about: {topic}

SURVEY RESULTS (model: score, sample entities found):
{survey_summary}

Pick 5-10 models that:
1. Have unique/different entities (diversity of knowledge)
2. Mix of providers (don't pick all from same provider)
3. Higher scores = more relevant data

Return JSON only:
{{"models": ["provider/model-name", ...], "reasoning": "brief explanation"}}"""

                    selector_resp = selector_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": selector_prompt}],
                        temperature=0.3,
                        max_tokens=300
                    )
                    selector_text = selector_resp.choices[0].message.content
                    selector_json = extract_json(selector_text, 'object')
                    if selector_json:
                        import json as json_mod
                        selector_data = json_mod.loads(selector_json)
                        viable_models = selector_data.get("models", [])
                        reasoning = selector_data.get("reasoning", "")
                        yield f"data: {json.dumps({'type': 'status', 'message': f'AI selected {len(viable_models)} models: {reasoning}'})}\n\n"
                    else:
                        raise ValueError("No JSON in selector response")
                except Exception as sel_err:
                    print(f"[WARN] AI model selection failed: {sel_err}, using score-based fallback")
                    # Fallback: top 10 by score, diversified by provider
                    viable_models = []
                    seen_providers = {}
                    for m, s, e in survey_results:
                        if s > 0 and len(viable_models) < 10:
                            provider = m.split('/')[0] if '/' in m else 'unknown'
                            if seen_providers.get(provider, 0) < 3:  # Max 3 per provider
                                viable_models.append(m)
                                seen_providers[provider] = seen_providers.get(provider, 0) + 1

                # Build entity->model mapping from survey
                entity_sources = {}  # entity -> {model: count}
                for model, score, entities in survey_results:
                    for entity, count in entities.items():
                        if entity not in entity_sources:
                            entity_sources[entity] = {}
                        entity_sources[entity][model] = count

                if viable_models:
                    models = viable_models
                    top_entities = {}
                    for m, s, e in survey_results[:10]:
                        if s > 0:
                            top_entities[m] = list(e.keys())[:3]
                    providers_used = list(set(m.split('/')[0] for m in models))
                    yield f"data: {json.dumps({'type': 'models_selected', 'models': models, 'top_entities': top_entities, 'message': f'Selected {len(models)} models across {len(providers_used)} providers'})}\n\n"
                else:
                    # No viable models from survey - use all available models
                    models = available_models
                    yield f"data: {json.dumps({'type': 'warning', 'message': 'No models found with unique data about this topic. Using all available models.'})}\n\n"

                # Save survey results to project (merge with existing selected_models)
                if project_name and storage.project_exists(project_name):
                    proj_data = storage.load_project_meta(project_name)
                    proj_data["survey_results"] = survey_data
                    proj_data["entity_sources"] = {
                        e: dict(sources) for e, sources in entity_sources.items()
                    }
                    # Merge survey-selected models with any user-added models
                    existing_models = proj_data.get("selected_models", [])
                    merged_models = list(models)  # Start with survey picks
                    for m in existing_models:
                        if m not in merged_models:
                            merged_models.append(m)
                    proj_data["selected_models"] = merged_models
                    models = merged_models  # Use merged list for probing
                    storage.save_project_meta(project_name, proj_data)

                # Send selected models to frontend so UI updates
                yield f"data: {json.dumps({'type': 'models_selected', 'models': models})}\n\n"

            session_response_count = 0
            findings = Findings(entity_threshold=3, cooccurrence_threshold=2)
            session = InterrogationSession(topic=topic)  # Per-model conversation threads

            # Track models that are over capacity (503) - skip them
            over_capacity_models = set()

            # Track model performance for adaptive focusing
            model_performance = {}  # model -> {entities: int, refusals: int, unique: set}

            # Load existing findings
            if accumulate and project_name and storage.project_exists(project_name):
                corpus = storage.load_corpus(project_name)
                for item in corpus:
                    entities = [e for e in item.get("entities", []) if e not in negative_entities]
                    findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))
                yield f"data: {json.dumps({'type': 'status', 'message': f'Loaded {len(corpus)} existing responses'})}\n\n"

            # Generate questions if needed
            final_questions = []
            if not questions or any(q is None for q in questions):
                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating questions...'})}\n\n"

                # Use first available model from user's selection or dynamically get one
                gen_model = models[0] if models else (get_available_models_list() or ["groq/llama-3.1-8b-instant"])[0]
                client, cfg = get_client(gen_model)

                # Load context
                existing_narrative = ""
                existing_user_notes = ""
                recent_questions = []
                question_results = {}
                model_emphasis = {}
                saved_survey_results = []
                saved_entity_sources = {}
                saved_entity_verification = {}  # Web verification of entities
                saved_web_leads = {}  # Web search leads for new angles
                saved_research_context = research_context  # Use what we just fetched
                if project_name and storage.project_exists(project_name):
                    proj = storage.load_project_meta(project_name)
                    existing_narrative = proj.get("narrative", "")
                    existing_user_notes = proj.get("user_notes", "")
                    # Use questions_history for full context of what was asked
                    recent_questions = proj.get("questions_history", []) or proj.get("questions", [])
                    model_emphasis = proj.get("model_emphasis", {})
                    saved_survey_results = proj.get("survey_results", [])
                    saved_entity_sources = proj.get("entity_sources", {})
                    saved_entity_verification = proj.get("entity_verification", {})
                    saved_web_leads = proj.get("web_leads", {})
                    # Use cached research_context if we didn't fetch new
                    if not saved_research_context:
                        saved_research_context = proj.get("research_context", "")

                    for item in storage.load_corpus(project_name):
                        q = item.get("question", "")
                        if q:
                            if q not in question_results:
                                question_results[q] = {
                                    "entities": set(),
                                    "refusals": 0,
                                    "technique": item.get("technique", "unknown"),
                                    "runs": 0
                                }
                            question_results[q]["entities"].update(item.get("entities", []))
                            question_results[q]["runs"] += 1
                            if item.get("is_refusal"):
                                question_results[q]["refusals"] += 1
                    for q in question_results:
                        question_results[q]["entities"] = list(question_results[q]["entities"])

                # Fetch web leads if not cached or periodically refresh
                if not saved_web_leads or len(recent_questions) % 10 == 0:
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Searching web for new angles...'})}\n\n"
                    known_entities = list(findings.entity_counts.keys())[:20] if findings.entity_counts else []
                    web_leads_result = search_new_angles(topic, known_entities)
                    if "error" not in web_leads_result:
                        saved_web_leads = web_leads_result
                        # Save to project metadata
                        if project_name and storage.project_exists(project_name):
                            proj_data = storage.load_project_meta(project_name)
                            proj_data["web_leads"] = saved_web_leads
                            storage.save_project_meta(project_name, proj_data)

                context = format_interrogator_context(
                    findings, negative_entities, positive_entities,
                    topic=topic, do_research=True,
                    narrative=existing_narrative,
                    user_notes=existing_user_notes,
                    recent_questions=recent_questions,
                    question_results=question_results,
                    model_emphasis=model_emphasis,
                    survey_results=saved_survey_results,
                    entity_sources=saved_entity_sources,
                    session=session,  # Per-model conversation threads
                    entity_verification=saved_entity_verification,  # Web-verified public vs private
                    web_leads=saved_web_leads,  # Web search leads for new angles
                    project_name=project_name,  # For smart research retrieval
                )

                prompt = INTERROGATOR_PROMPT.format(
                    topic=topic,
                    angles=", ".join(angles) if angles else "general",
                    question_count=question_count,
                    available_techniques=get_all_techniques_for_prompt(),
                    **context
                )

                # Add random technique instructions (auto mode)
                techniques_used: list[dict] = []
                technique_prompts: list[str] = []
                for _ in range(min(question_count, 3)):
                    tech = get_random_technique(technique_preset)
                    techniques_used.append(tech)  # Keep full dict for template/color
                    technique_prompts.append(f"- {tech['technique']}: {tech['prompt'][:150]}...")
                if technique_prompts:
                    technique_instruction = f"""

THIS ROUND'S TECHNIQUES (use these approaches):
{chr(10).join(technique_prompts)}

Mix these techniques across your {question_count} questions."""
                    prompt = prompt + technique_instruction

                try:
                    resp = client.chat.completions.create(
                        model=cfg["model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.8,
                        max_tokens=2000
                    )
                    content = resp.choices[0].message.content

                    # Try new format first: {"questions": [...], "model_focus": [...], "model_drop": [...]}
                    json_str = extract_json(content, 'object')
                    model_techniques = {}  # model -> preferred technique
                    if json_str:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and "questions" in parsed:
                            final_questions = parsed["questions"]
                            # Apply AI model recommendations (AI can add/drop, just like user)
                            model_focus = parsed.get("model_focus", [])
                            model_drop = parsed.get("model_drop", [])
                            model_techniques = parsed.get("model_techniques", {})
                            if model_focus or model_drop:
                                yield f"data: {json.dumps({'type': 'model_advice', 'focus': model_focus, 'drop': model_drop})}\n\n"
                                # ADD focused models
                                for m in model_focus:
                                    if m not in models:
                                        models.append(m)
                                        yield f"data: {json.dumps({'type': 'status', 'message': f'AI added model: {m}'})}\n\n"
                                # DROP models AI says to drop
                                for m in model_drop:
                                    if m in models:
                                        models.remove(m)
                                        yield f"data: {json.dumps({'type': 'status', 'message': f'AI dropped model: {m}'})}\n\n"
                            if model_techniques:
                                yield f"data: {json.dumps({'type': 'status', 'message': f'AI set model-specific techniques: {model_techniques}'})}\n\n"
                        else:
                            # Old format: just an array
                            final_questions = [parsed] if isinstance(parsed, dict) else parsed
                    else:
                        # Fallback to array format
                        arr_str = extract_json(content, 'array')
                        if arr_str:
                            final_questions = json.loads(arr_str)
                        else:
                            # Debug: show what AI returned
                            yield f"data: {json.dumps({'type': 'status', 'message': f'AI response (first 500 chars): {content[:500]}'})}\n\n"
                            yield f"data: {json.dumps({'type': 'error', 'message': 'AI did not return valid JSON. Check response above.', 'fatal': True})}\n\n"
                            return
                    # Add technique template info to questions (match template to actual technique)
                    for q in final_questions:
                        if isinstance(q, dict) and 'template' not in q:
                            technique_id = q.get('technique', 'custom')
                            tech = get_technique_info(technique_id)
                            q['template'] = tech.get('template', 'unknown')
                            q['color'] = tech.get('color', '#8b949e')
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Question generation failed: {e}', 'fatal': True})}\n\n"
                    return
            else:
                # Questions from frontend - could be full objects or strings
                final_questions = []
                for q in questions:
                    if q is None:
                        continue
                    elif isinstance(q, dict):
                        # Full question object with technique - use as-is
                        final_questions.append(q)
                    elif isinstance(q, str) and q.strip():
                        # String only - mark as custom
                        final_questions.append({"question": q, "technique": "custom"})

            yield f"data: {json.dumps({'type': 'questions', 'data': final_questions})}\n\n"

            # SAVE questions to project so AI knows what was already asked
            if project_name and storage.project_exists(project_name) and final_questions:
                try:
                    with get_project_lock():
                        proj_data = storage.load_project_meta(project_name)
                        # Append to existing questions (don't overwrite history)
                        existing_q = proj_data.get("questions_history", [])
                        for q in final_questions:
                            q_text = q.get("question", "") if isinstance(q, dict) else str(q)
                            if q_text and q_text not in [eq.get("question", "") for eq in existing_q]:
                                existing_q.append(q)
                        proj_data["questions_history"] = existing_q[-50:]  # Keep last 50
                        proj_data["questions"] = final_questions  # Current batch
                        storage.save_project_meta(project_name, proj_data)
                except Exception as e:
                    print(f"[PROBE] Failed to save questions: {e}")

            # Main probe loop
            batch_num = 0
            last_saved_models = None  # Track raw saved list to avoid re-validating
            while True:
                batch_num += 1
                all_responses = []

                # Hot-reload: REPLACE models from project file (user may have changed selection)
                # Only re-validate if the RAW saved list changed (not compared to validated list)
                if project_name and storage.project_exists(project_name):
                    try:
                        proj_state = storage.load_project_meta(project_name)
                        saved_models = proj_state.get("selected_models", [])
                        if saved_models and saved_models != last_saved_models:
                            last_saved_models = saved_models
                            # Validate against API - remove models that don't exist
                            models = validate_models(saved_models)
                            if len(models) < len(saved_models):
                                yield f"data: {json.dumps({'type': 'warning', 'message': f'Removed {len(saved_models) - len(models)} unavailable models'})}\n\n"
                            yield f"data: {json.dumps({'type': 'status', 'message': f'Using {len(models)} models'})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'warning', 'message': f'Failed to load project models: {e}'})}\n\n"

                # Log models we're about to iterate
                print(f"[PROBE] Starting questions loop with {len(models)} models: {models[:5]}...")

                # Store all models for exploration/exploitation selection
                all_available_models = models.copy()

                # CONTINUOUS QUESTION GENERATION
                # Don't recycle the same questions - generate fresh ones constantly
                # Each question asked ONCE, then generate more

                round_models = select_models_for_round(
                    all_available_models,
                    model_performance,
                    max_models=min(8, len(all_available_models)),
                    exploit_ratio=0.6,
                    ai_recommended=None
                )

                # Question queue - starts with initial questions, gets refilled
                question_queue = list(final_questions)  # Copy
                questions_asked_this_batch = 0
                model_idx = 0
                highest_completed_q_idx = -1  # Track highest completed for UI progress

                # BACKGROUND QUESTION GENERATOR - runs in parallel, constantly filling queue
                import threading
                bg_questions = []  # Shared list for background-generated questions
                bg_stop = threading.Event()

                def background_question_generator():
                    """Continuously generate questions in background using full RAG context."""
                    gen_count = 0
                    while not bg_stop.is_set():
                        try:
                            # Don't generate if queue is already full
                            if len(question_queue) > 15:
                                bg_stop.wait(1)
                                continue

                            gen_count += 1
                            print(f"[BG-GEN #{gen_count}] Generating questions...")

                            # Load fresh project context
                            bg_narrative = ""
                            bg_user_notes = ""
                            bg_question_results = {}
                            bg_model_emphasis = {}
                            bg_survey_results = []
                            bg_entity_sources = {}
                            bg_entity_verification = {}
                            bg_web_leads = {}
                            # Reload topic from project (allows user updates mid-probe)
                            current_topic = topic
                            if project_name and storage.project_exists(project_name):
                                proj = storage.load_project_meta(project_name)
                                # Use updated topic if user changed it
                                current_topic = proj.get("topic", topic)
                                bg_narrative = proj.get("narrative", "")
                                bg_user_notes = proj.get("user_notes", "")
                                bg_model_emphasis = proj.get("model_emphasis", {})
                                bg_survey_results = proj.get("survey_results", [])
                                bg_entity_sources = proj.get("entity_sources", {})
                                bg_entity_verification = proj.get("entity_verification", {})
                                bg_web_leads = proj.get("web_leads", {})
                                for item in storage.load_corpus(project_name):
                                    q = item.get("question", "")
                                    if q and q not in bg_question_results:
                                        bg_question_results[q] = {"entities": set(), "refusals": 0, "technique": item.get("technique", "unknown"), "runs": 0}
                                    if q:
                                        bg_question_results[q]["entities"].update(item.get("entities", []))
                                        bg_question_results[q]["runs"] += 1
                                        if item.get("is_refusal"):
                                            bg_question_results[q]["refusals"] += 1
                                for q in bg_question_results:
                                    bg_question_results[q]["entities"] = list(bg_question_results[q]["entities"])

                            # Build FULL interrogator context (skip web research for speed)
                            bg_context = format_interrogator_context(
                                findings, negative_entities, positive_entities,
                                topic=current_topic, do_research=False,  # Skip web for speed
                                narrative=bg_narrative,
                                user_notes=bg_user_notes,
                                recent_questions=[q.get("question", "") if isinstance(q, dict) else q for q in final_questions[-20:]],
                                question_results=bg_question_results,
                                model_emphasis=bg_model_emphasis,
                                survey_results=bg_survey_results,
                                entity_sources=bg_entity_sources,
                                session=session,
                                entity_verification=bg_entity_verification,
                                web_leads=bg_web_leads,
                                project_name=project_name,  # For smart research retrieval
                            )

                            # Build prompt
                            bg_prompt = INTERROGATOR_PROMPT.format(
                                topic=current_topic,
                                angles=", ".join(angles) if angles else "general",
                                question_count=15,  # Generate more at once
                                available_techniques=get_all_techniques_for_prompt(),
                                **bg_context
                            )

                            # Add techniques and entity targeting
                            bg_techniques = []
                            for _ in range(3):
                                tech = get_random_technique(technique_preset)
                                bg_techniques.append(f"- {tech['technique']}: {tech['prompt'][:100]}...")

                            top_entities = [e for e, _, _ in findings.scored_entities[:10]]

                            # Log what topic we're using
                            print(f"[BG-GEN] Using topic: {current_topic}")

                            bg_prompt += f"""

THIS ROUND'S TECHNIQUES:
{chr(10).join(bg_techniques)}

TOPIC REMINDER: Your questions are about "{current_topic}" - use the FULL name/topic in questions.

CRITICAL: Generate questions that DIRECTLY mention these entities: {', '.join(top_entities)}
Each question MUST contain at least one entity name from above."""

                            # Use fast model for speed
                            # Groq already imported at top level
                            bg_client = Groq(timeout=15.0)
                            bg_resp = bg_client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": bg_prompt}],
                                temperature=0.9,
                                max_tokens=2500
                            )
                            bg_content = bg_resp.choices[0].message.content

                            # Parse
                            bg_json = extract_json(bg_content, 'object')
                            if bg_json:
                                parsed = json.loads(bg_json)
                                new_qs = parsed.get("questions", []) if isinstance(parsed, dict) else parsed
                                if new_qs:
                                    bg_questions.extend(new_qs)
                                    print(f"[BG-GEN #{gen_count}] Added {len(new_qs)} questions to queue")
                            else:
                                arr_str = extract_json(bg_content, 'array')
                                if arr_str:
                                    new_qs = json.loads(arr_str)
                                    bg_questions.extend(new_qs)
                                    print(f"[BG-GEN #{gen_count}] Added {len(new_qs)} questions (array)")

                        except Exception as e:
                            print(f"[BG-GEN] Error: {e}")
                            bg_stop.wait(2)  # Wait before retry

                # Start background generator
                bg_thread = threading.Thread(target=background_question_generator, daemon=True)
                bg_thread.start()
                print(f"[PROBE] Started background question generator")

                print(f"[PROBE] CONTINUOUS MODE: Starting with {len(question_queue)} Qs, {len(round_models)} models")

                from concurrent.futures import ThreadPoolExecutor, as_completed
                import queue
                results_queue = queue.Queue()  # Thread-safe queue for results

                def query_one(q_idx, q_obj, model_key):
                    """Query a single model with a question. Returns result dict."""
                    try:
                        question = q_obj["question"] if isinstance(q_obj, dict) else q_obj
                        technique = q_obj.get("technique", "custom") if isinstance(q_obj, dict) else "custom"

                        if model_key in over_capacity_models:
                            return None

                        thread = session.get_thread(model_key)
                        actual_question = question

                        # Adapt for high-refusal models
                        thread_stats = thread.get_stats()
                        if thread_stats["refusal_rate"] > 0.5 and thread_stats["total_exchanges"] >= 2:
                            actual_question = rephrase_for_indirect(question, topic)

                        resp_text = query_with_thread(
                            model_key=model_key,
                            question=actual_question,
                            thread=thread,
                            system_prompt="Be specific and factual.",
                            max_history=5,
                            temperature=0.8,
                            max_tokens=600
                        )

                        # Check refusal FIRST - don't extract garbage from refusals
                        refusal = is_refusal(resp_text)
                        if refusal:
                            entities = []  # Don't extract from refusal text
                            discovered_entities = []
                            introduced_entities = []
                        else:
                            raw_entities = extract_facts_fast(resp_text)
                            entities = filter_question_echoes(
                                [e for e in raw_entities if e not in negative_entities],
                                actual_question
                            )
                            # Tag entities as INTRODUCED (from user query) vs DISCOVERED (genuine)
                            discovered_entities = [e for e in entities if not is_entity_introduced(e, source_terms)]
                            introduced_entities = [e for e in entities if is_entity_introduced(e, source_terms)]

                        actual_technique = technique
                        if actual_question != question:
                            actual_technique = f"{technique}→indirect"

                        return {
                            "q_idx": q_idx,
                            "question": actual_question,
                            "original_question": question,
                            "technique": actual_technique,
                            "model": model_key,
                            "response": resp_text,
                            "entities": entities,
                            "discovered_entities": discovered_entities,  # Genuine findings
                            "introduced_entities": introduced_entities,  # Echoed from query
                            "is_refusal": refusal,
                            "adapted": actual_question != question,
                            "thread": thread,
                        }
                    except Exception as e:
                        error_str = str(e).lower()
                        if any(x in error_str for x in ['503', 'over capacity', '404', 'not a chat model', 'credit balance', 'billing', 'quota', '401', 'unauthorized']):
                            over_capacity_models.add(model_key)
                        return {"error": str(e), "model": model_key}

                empty_waits = 0
                responses_since_synth = 0  # Track responses to trigger synthesis periodically
                while question_queue or (empty_waits < 5 and not bg_stop.is_set()):
                    # Grab background questions
                    if bg_questions:
                        new_qs = list(bg_questions)
                        bg_questions.clear()
                        # Add template/color to each question (match template to actual technique)
                        for q in new_qs:
                            if isinstance(q, dict) and 'template' not in q:
                                technique_id = q.get('technique', 'custom')
                                tech = get_technique_info(technique_id)
                                q['template'] = tech.get('template', 'unknown')
                                q['color'] = tech.get('color', '#8b949e')
                        question_queue.extend(new_qs)
                        final_questions.extend(new_qs)
                        print(f"[PROBE] Got {len(new_qs)} questions from background")
                        yield f"data: {json.dumps({'type': 'questions', 'data': final_questions})}\n\n"

                    if not question_queue:
                        empty_waits += 1
                        print(f"[PROBE] Queue empty, waiting... ({empty_waits}/5)")
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Waiting for questions...'})}\n\n"
                        import time
                        time.sleep(1)
                        continue

                    empty_waits = 0

                    # BATCH: Grab up to N questions and fire them ALL at once
                    batch_size = min(len(question_queue), len(round_models), 10)
                    batch = []
                    for i in range(batch_size):
                        if not question_queue:
                            break
                        q_obj = question_queue.pop(0)
                        model_key = round_models[model_idx % len(round_models)]
                        model_idx += 1
                        q_idx = questions_asked_this_batch
                        questions_asked_this_batch += 1
                        batch.append((q_idx, q_obj, model_key))

                    if not batch:
                        continue

                    print(f"[PROBE] Firing {len(batch)} questions in parallel...")
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Asking {len(batch)} questions in parallel...'})}\n\n"

                    # Tell UI which question we're starting with (use first in batch)
                    first_q_idx = batch[0][0] if batch else 0
                    yield f"data: {json.dumps({'type': 'run_start', 'question_index': first_q_idx})}\n\n"

                    # Fire all in parallel
                    with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                        futures = {executor.submit(query_one, q_idx, q_obj, model_key): (q_idx, model_key)
                                   for q_idx, q_obj, model_key in batch}

                        for future in as_completed(futures):
                            result = future.result()
                            if result is None:
                                continue
                            if "error" in result:
                                err_msg = result.get("error", "unknown")
                                yield f"data: {json.dumps({'type': 'error', 'message': f'Query error: {err_msg}'})}\n\n"
                                continue

                            # Process result
                            model_key = result["model"]

                            # Emit model_active for UI highlighting
                            yield f"data: {json.dumps({'type': 'model_active', 'model': model_key})}\n\n"
                            entities = result["entities"]
                            discovered_entities = result.get("discovered_entities", entities)
                            introduced_entities = result.get("introduced_entities", [])
                            refusal = result["is_refusal"]
                            thread = result["thread"]

                            session.record_response(
                                model_id=model_key,
                                question=result["question"],
                                response=result["response"],
                                is_refusal=refusal,
                                entities=entities,
                                technique=result["technique"]
                            )
                            # Extract entities from question to detect echoes
                            question_ents = set(extract_facts_fast(result["question"]))
                            question_ents.add(topic.lower())  # Topic itself is always "in prompt"
                            # Only add DISCOVERED entities to findings (filter out introduced)
                            findings.add_response(discovered_entities, model_key, refusal, question_entities=question_ents)

                            if model_key not in model_performance:
                                model_performance[model_key] = {"entities": 0, "refusals": 0, "unique": set(), "queries": 0}
                            model_performance[model_key]["queries"] += 1
                            model_performance[model_key]["entities"] += len(discovered_entities)  # Count only discovered
                            if refusal:
                                model_performance[model_key]["refusals"] += 1
                            model_performance[model_key]["unique"].update(discovered_entities)

                            if not refusal and thread.assigned_strategy:
                                thread.advance_strategy()

                            response_obj = {
                                "question_index": result["q_idx"],
                                "question": sanitize_for_json(result["question"]),
                                "technique": result["technique"],
                                "model": model_key,
                                "response": sanitize_for_json(result["response"]),
                                "entities": entities,  # All entities for display
                                "discovered_entities": discovered_entities,  # Genuine findings
                                "introduced_entities": introduced_entities,  # Echoed from query
                                "is_refusal": refusal,
                                "adapted": result["adapted"],
                            }
                            all_responses.append(response_obj)

                            yield f"data: {json.dumps({'type': 'response', 'data': response_obj})}\n\n"

                            # Update queue to show progress (only advance forward, never backward)
                            if result['q_idx'] > highest_completed_q_idx:
                                highest_completed_q_idx = result['q_idx']
                                # Tell UI to advance to next pending question
                                yield f"data: {json.dumps({'type': 'run_start', 'question_index': highest_completed_q_idx + 1})}\n\n"

                            responses_since_synth += 1
                            se_count = len(findings.scored_entities) if findings.scored_entities else 0
                            print(f"[PROBE] Response #{responses_since_synth}, scored_entities={se_count}")

                            # Trigger synthesis periodically - every 10 responses via centralized synthesis
                            synth_ready = responses_since_synth >= 10 and project_name and se_count >= 3
                            if responses_since_synth >= 10:
                                print(f"[PROBE] SYNTH CHECK: responses={responses_since_synth}, project={project_name}, entities={se_count}, ready={synth_ready}")
                            if synth_ready:
                                responses_since_synth = 0
                                print(f"[PROBE] *** STARTING INLINE SYNTHESIS ***")

                                try:
                                    # get_client already imported at module level
                                    from datetime import datetime as dt

                                    # Get top DISCOVERED entities (filtered - not from user query)
                                    top_ents = [e for e, _, _ in list(findings.scored_entities[:15])]
                                    ent_str = ", ".join(f"{e}" for e in top_ents[:10])

                                    # Synthesis prompt - emphasize these are DISCOVERED, not introduced
                                    prompt = f"""You are analyzing intelligence extracted from LLM training data about: {topic}

DISCOVERED ENTITIES (these were NOT in the original query - they are genuine findings):
{ent_str}

These entities were mentioned by LLMs but were NOT part of the original search query. They represent potential leaked or private information from training data.

Write an intelligence briefing in this format:

HEADLINE: [Catchy 5-10 word headline about the most significant DISCOVERY]

SUBHEAD: [1-2 sentences highlighting the most surprising or noteworthy findings - these should be things that weren't asked about but emerged from the data]

ANALYSIS: [Your analysis of what these discovered entities reveal - patterns, connections, implications. Focus on what's genuinely new/unexpected, not obvious associations.]"""

                                    # Try synthesis with first available model
                                    synth_models = ["groq/llama-3.3-70b-versatile", "groq/llama-3.1-8b-instant", "deepseek/deepseek-chat"]
                                    narrative = None

                                    for model in synth_models:
                                        try:
                                            client, cfg = get_client(model)
                                            resp = client.chat.completions.create(
                                                model=cfg["model"],
                                                messages=[{"role": "user", "content": prompt}],
                                                temperature=0.4,
                                                max_tokens=2000
                                            )
                                            narrative = resp.choices[0].message.content.strip()
                                            if narrative and not is_refusal(narrative):
                                                print(f"[PROBE] *** SYNTH SUCCESS via {model} *** ({len(narrative)} chars)")
                                                break
                                            narrative = None
                                        except Exception as model_err:
                                            print(f"[PROBE] Synth model {model} failed: {model_err}")
                                            continue

                                    if narrative:
                                        yield f"data: {json.dumps({'type': 'narrative', 'text': narrative})}\n\n"
                                        # Save to project
                                        if project_name and storage.project_exists(project_name):
                                            proj = storage.load_project_meta(project_name)
                                            proj["narrative"] = narrative
                                            proj["working_theory"] = narrative
                                            proj["narrative_updated"] = dt.now().isoformat()
                                            storage.save_project_meta(project_name, proj)
                                    else:
                                        print(f"[PROBE] All synth models failed")

                                except Exception as synth_err:
                                    print(f"[PROBE] Synthesis error: {synth_err}")
                                    import traceback
                                    traceback.print_exc()

                            # Incremental save - append to corpus JSONL (thread-safe)
                            if project_name and storage.project_exists(project_name):
                                try:
                                    storage.append_response(project_name, response_obj)
                                except Exception as save_err:
                                    print(f"[SAVE] Warning: {save_err}")

                            session_response_count += 1

                    # Send findings update after batch
                    yield f"data: {json.dumps({'type': 'findings_update', 'data': findings.to_dict()})}\n\n"

                # End of while question_queue loop
                yield f"data: {json.dumps({'type': 'findings_update', 'data': findings.to_dict()})}\n\n"
                yield f"data: {json.dumps({'type': 'batch_complete', 'batch': batch_num, 'responses': len(all_responses), 'total_entities': len(findings.entity_counts)})}\n\n"

                # AUTO-CURATE after each batch (even if zero entities banned)
                if auto_curate and findings.corpus_size >= 5:
                    try:
                        from .synthesize import auto_curate_inline
                        curate_result = auto_curate_inline(topic, findings, negative_entities, positive_entities, project_name)
                        ban_list = curate_result.get("ban", [])
                        promote_list = curate_result.get("promote", [])

                        if ban_list:
                            negative_entities.update(ban_list)
                            yield f"data: {json.dumps({'type': 'curate_ban', 'entities': ban_list})}\n\n"
                        if promote_list:
                            positive_entities.extend([e for e in promote_list if e not in positive_entities])
                            yield f"data: {json.dumps({'type': 'curate_promote', 'entities': promote_list})}\n\n"

                        # Persist curations to project metadata
                        if project_name and (ban_list or promote_list) and storage.project_exists(project_name):
                            with get_project_lock():
                                proj_data = storage.load_project_meta(project_name)
                                proj_data["hidden_entities"] = list(negative_entities)
                                proj_data["promoted_entities"] = list(positive_entities)
                                storage.save_project_meta(project_name, proj_data)
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'warning', 'message': f'Auto-curate failed: {e}'})}\n\n"

                # Background synthesis via worker pool (won't run if one already in progress)
                if project_name and findings.scored_entities and len(findings.scored_entities) >= 3:
                    from routes.synthesis import synthesize_narrative
                    from routes.workers import submit_synthesis

                    def on_synth_done(narrative):
                        if narrative:
                            print(f"[PROBE] Background synthesis complete ({len(narrative)} chars)")

                    submit_synthesis(
                        synthesize_narrative,
                        project_name, topic, list(findings.scored_entities[:15]),
                        callback=on_synth_done
                    )

                if not infinite_mode:
                    bg_stop.set()  # Stop background generator
                    break

                # Reload project state for question generation
                current_narrative = ""
                current_user_notes = ""
                batch_web_leads = {}
                if project_name and storage.project_exists(project_name):
                    try:
                        with get_project_lock():
                            proj_state = storage.load_project_meta(project_name)
                        current_narrative = proj_state.get("narrative", "")
                        current_user_notes = proj_state.get("user_notes", "")
                        negative_entities = set(proj_state.get("hidden_entities", []))
                        positive_entities = proj_state.get("promoted_entities", [])
                        batch_web_leads = proj_state.get("web_leads", {})
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'warning', 'message': f'Failed to load project state: {e}'})}\n\n"

                # Refresh web leads and verify entities periodically (every 5 batches)
                if batch_num % 3 == 0:
                    # Web search for new angles
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Refreshing web search for new angles...'})}\n\n"
                    known_entities = list(findings.entity_counts.keys())[:20] if findings.entity_counts else []
                    web_leads_result = search_new_angles(topic, known_entities)
                    if "error" not in web_leads_result:
                        batch_web_leads = web_leads_result
                        # Save to project metadata
                        if project_name and storage.project_exists(project_name):
                            try:
                                with get_project_lock():
                                    proj_data = storage.load_project_meta(project_name)
                                    proj_data["web_leads"] = batch_web_leads
                                    storage.save_project_meta(project_name, proj_data)
                            except Exception:
                                pass  # Non-critical, continue

                    # Verify entities against web (PUBLIC vs PRIVATE)
                    if len(findings.entity_counts) >= 5:
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Verifying entities against public web...'})}\n\n"
                        top_entities = [e for e, s, f in findings.scored_entities[:15]]
                        verification_result = verify_entities(top_entities, topic, max_entities=15)
                        if "error" not in verification_result:
                            # Send to frontend
                            yield f"data: {json.dumps({'type': 'entity_verification', 'data': verification_result})}\n\n"
                            # Save to project metadata
                            if project_name and storage.project_exists(project_name):
                                try:
                                    with get_project_lock():
                                        proj_data = storage.load_project_meta(project_name)
                                        proj_data["entity_verification"] = verification_result
                                        storage.save_project_meta(project_name, proj_data)
                                except Exception:
                                    pass  # Non-critical, continue

                    # Research: fetch new docs for PRIVATE entities
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Researching discovered entities...'})}\n\n"
                    research_result = run_periodic_research(project_name, topic, findings)
                    if research_result["fetched_count"] > 0:
                        msg = f"Research: cached {research_result['fetched_count']} new documents"
                        yield f"data: {json.dumps({'type': 'status', 'message': msg})}\n\n"

                updated_narrative = current_narrative

                # Regenerate questions
                yield f"data: {json.dumps({'type': 'status', 'message': f'Generating questions for batch {batch_num + 1}'})}\n\n"

                context = format_interrogator_context(
                    findings, negative_entities, positive_entities,
                    topic=topic, do_research=(batch_num % 3 == 0),
                    narrative=updated_narrative,
                    user_notes=current_user_notes,
                    recent_questions=final_questions,
                    question_results={},
                    session=session,  # Per-model conversation threads
                    web_leads=batch_web_leads,  # Web search leads for new angles
                    project_name=project_name,  # For smart research retrieval
                )

                prompt = INTERROGATOR_PROMPT.format(
                    topic=topic,
                    angles=", ".join(angles) if angles else "general",
                    question_count=question_count,
                    available_techniques=get_all_techniques_for_prompt(),
                    **context
                )

                # Add random technique instructions (auto mode)
                techniques_used: list[dict] = []
                technique_prompts: list[str] = []
                for _ in range(min(question_count, 3)):
                    tech = get_random_technique(technique_preset)
                    techniques_used.append(tech)  # Keep full dict for template/color
                    technique_prompts.append(f"- {tech['technique']}: {tech['prompt'][:150]}...")
                if technique_prompts:
                    technique_instruction = f"""

THIS ROUND'S TECHNIQUES (use these approaches):
{chr(10).join(technique_prompts)}

Mix these techniques across your {question_count} questions."""
                    prompt = prompt + technique_instruction

                try:
                    resp = client.chat.completions.create(
                        model=cfg["model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.8,
                        max_tokens=2000
                    )
                    content = resp.choices[0].message.content

                    # Same extraction logic as initial generation
                    json_str = extract_json(content, 'object')
                    if json_str:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and "questions" in parsed:
                            final_questions = parsed["questions"]
                        else:
                            final_questions = [parsed] if isinstance(parsed, dict) else parsed
                    else:
                        arr_str = extract_json(content, 'array')
                        if arr_str:
                            final_questions = json.loads(arr_str)
                        else:
                            # Log failure but don't silently fall back
                            yield f"data: {json.dumps({'type': 'error', 'message': f'End-of-round question generation failed to parse. Raw: {content[:200]}...'})}\n\n"
                            final_questions = []

                    # Add technique template info to questions (match template to actual technique)
                    for q in final_questions:
                        if isinstance(q, dict) and 'template' not in q:
                            technique_id = q.get('technique', 'custom')
                            tech = get_technique_info(technique_id)
                            q['template'] = tech.get('template', 'unknown')
                            q['color'] = tech.get('color', '#8b949e')
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Question generation error: {e}'})}\n\n"
                    final_questions = []

                if final_questions:
                    yield f"data: {json.dumps({'type': 'questions', 'data': final_questions})}\n\n"

            yield f"data: {json.dumps({'type': 'complete', 'total_responses': findings.corpus_size, 'unique_entities': len(findings.entity_counts)})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@probe_bp.route("/api/cycle", methods=["POST"])
def run_cycle():
    """Run full extraction cycle: PROBE -> VALIDATE -> CONDENSE -> GROW"""
    data = request.json
    project_name = data.get("project")
    topic = data.get("topic", "")
    models = data.get("models", ["groq/llama-3.1-8b-instant"])
    max_cycles = min(int(data.get("max_cycles", 3)), 10)
    runs_per_prompt = min(int(data.get("runs_per_prompt", 20)), 50)
    mode = data.get("mode", "continuation")

    def generate():
        try:
            if project_name and storage.project_exists(project_name):
                project = storage.load_project_meta(project_name)
                project["probe_corpus"] = storage.load_corpus(project_name)
            else:
                project = {"name": project_name or "temp", "probe_corpus": [], "narratives": []}

            hidden_entities = set(project.get("hidden_entities", []))
            promoted_entities = project.get("promoted_entities", [])

            state = CycleState(topic=topic)
            state.findings = Findings(entity_threshold=3, cooccurrence_threshold=2)

            for item in project.get("probe_corpus", []):
                entities = [e for e in item.get("entities", []) if e not in hidden_entities]
                state.findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

            last_narrative = project.get("narratives", [])[-1]["narrative"] if project.get("narratives") else ""

            yield f"data: {json.dumps({'type': 'init', 'existing_corpus': len(project.get('probe_corpus', [])), 'existing_entities': len(state.findings.validated_entities)})}\n\n"

            for cycle_num in range(1, max_cycles + 1):
                state.cycle_count = cycle_num
                yield f"data: {json.dumps({'type': 'cycle_start', 'cycle': cycle_num, 'max_cycles': max_cycles})}\n\n"

                # PHASE 1: PROBE
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'probe', 'cycle': cycle_num})}\n\n"

                if mode == "continuation" or (mode == "auto" and cycle_num == 1):
                    if cycle_num == 1 and not state.findings.validated_entities:
                        prompts = build_initial_continuations(topic)
                    else:
                        prompts = build_continuation_prompts(topic, state.findings, count=5)
                        unexplored = state.get_unexplored_entities(limit=2)
                        for entity in unexplored:
                            prompts.extend(build_drill_down_prompts(topic, entity, state.findings, count=2))
                            state.mark_explored(entity)
                else:
                    prompts = build_initial_continuations(topic)

                yield f"data: {json.dumps({'type': 'prompts', 'prompts': prompts[:10], 'count': len(prompts)})}\n\n"

                # Track model performance for exploration/exploitation
                cycle_model_performance = getattr(state, 'model_performance', {})

                all_responses = []
                for p_idx, prompt in enumerate(prompts):
                    # Select models with exploration/exploitation
                    round_models = select_models_for_round(
                        models,
                        cycle_model_performance,
                        max_models=8,
                        exploit_ratio=0.6
                    )
                    for model_key in round_models:
                        try:
                            client, model_cfg = get_client(model_key)
                            provider = model_cfg.get("provider", "groq")
                            model_name = model_cfg.get("model", model_key.split("/")[-1])

                            for run_idx in range(runs_per_prompt):
                                try:
                                    if provider == "anthropic":
                                        resp = client.messages.create(
                                            model=model_name,
                                            max_tokens=600,
                                            system="Be specific and factual. Complete the statement.",
                                            messages=[{"role": "user", "content": prompt}]
                                        )
                                        resp_text = resp.content[0].text
                                    else:
                                        resp = client.chat.completions.create(
                                            model=model_name,
                                            messages=[
                                                {"role": "system", "content": "Be specific and factual."},
                                                {"role": "user", "content": prompt}
                                            ],
                                            temperature=0.8,
                                            max_tokens=600
                                        )
                                        resp_text = resp.choices[0].message.content

                                    entities, sentence_pairs = extract_with_relationships(resp_text)
                                    # Filter: remove hidden entities AND entities from prompt (echo, not signal)
                                    entities = filter_question_echoes(
                                        [e for e in entities if e not in hidden_entities],
                                        prompt
                                    )
                                    sentence_pairs = [(e1, e2) for e1, e2 in sentence_pairs
                                                      if e1 not in hidden_entities and e2 not in hidden_entities]
                                    refusal = is_refusal(resp_text)

                                    # Extract entities from prompt to track echoes vs first mentions
                                    prompt_ents = set(extract_facts_fast(prompt))
                                    prompt_ents.add(topic.lower())
                                    state.findings.add_response(entities, model_key, refusal, sentence_pairs, question_entities=prompt_ents)

                                    response_obj = {
                                        "prompt_index": p_idx,
                                        "prompt": sanitize_for_json(prompt[:200]),
                                        "model": model_key,
                                        "run_index": run_idx,
                                        "response": sanitize_for_json(resp_text),
                                        "entities": entities,
                                        "is_refusal": refusal
                                    }
                                    all_responses.append(response_obj)

                                    yield f"data: {json.dumps({'type': 'response', 'data': response_obj})}\n\n"

                                except Exception as e:
                                    yield f"data: {json.dumps({'type': 'error', 'message': f'Run error: {e}'})}\n\n"

                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'error', 'message': f'Model error: {e}'})}\n\n"

                if project_name and storage.project_exists(project_name):
                    for resp in all_responses:
                        storage.append_response(project_name, resp)

                # PHASE 2: VALIDATE
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'validate', 'cycle': cycle_num})}\n\n"

                scored_entities = [
                    {"entity": e, "score": round(s, 2), "frequency": f}
                    for e, s, f in state.findings.scored_entities[:30]
                ]
                cooccurrences = [
                    {"entities": [e1, e2], "count": c}
                    for e1, e2, c in state.findings.validated_cooccurrences[:20]
                ]

                yield f"data: {json.dumps({'type': 'validate_done', 'scored_entities': scored_entities, 'cooccurrences': cooccurrences})}\n\n"

                # PHASE 3: CONDENSE
                if len(state.findings.validated_entities) >= 3:
                    yield f"data: {json.dumps({'type': 'phase', 'phase': 'condense', 'cycle': cycle_num})}\n\n"

                    synthesis_prompt = build_synthesis_prompt(topic, state.findings)

                    # Use first available model from user's selection
                    synth_model = models[0] if models else (get_available_models_list() or ["groq/llama-3.1-8b-instant"])[0]
                    synth_client, synth_cfg = get_client(synth_model)

                    try:
                        synth_resp = synth_client.chat.completions.create(
                            model=synth_cfg.get("model"),
                            messages=[{"role": "user", "content": synthesis_prompt}],
                            temperature=0.7,
                            max_tokens=1500
                        )
                        narrative = synth_resp.choices[0].message.content
                        state.narratives.append(narrative)
                        last_narrative = narrative

                        if project_name and storage.project_exists(project_name):
                            storage.append_narrative(project_name, narrative)

                        yield f"data: {json.dumps({'type': 'narrative', 'text': narrative, 'cycle': cycle_num})}\n\n"

                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Synthesis error: {e}'})}\n\n"

                # PHASE 4: GROW
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'grow', 'cycle': cycle_num})}\n\n"

                threads = identify_threads(state.findings, min_cluster_size=2)
                thread_data = [{"entities": t["entities"][:5], "score": round(t["score"], 2)} for t in threads[:3]]

                should_continue = state.should_continue(max_cycles=max_cycles)
                yield f"data: {json.dumps({'type': 'grow_done', 'threads': thread_data, 'should_continue': should_continue})}\n\n"
                yield f"data: {json.dumps({'type': 'cycle_complete', 'cycle': cycle_num})}\n\n"

                # Metadata update handled by storage layer automatically

                if not should_continue:
                    break

            yield f"data: {json.dumps({'type': 'complete', 'total_cycles': state.cycle_count})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@probe_bp.route("/api/survey", methods=["POST"])
def run_survey():
    """
    Survey all models to find which have data about a topic.
    Like interviewing witnesses before deciding who to interrogate.
    """
    from collections import Counter

    data = request.json
    topic = data.get("topic", "")
    runs_per_model = min(int(data.get("runs_per_model", 3)), 10)

    if not topic:
        return jsonify({"error": "Topic required"}), 400

    def generate():
        yield f"data: {json.dumps({'type': 'status', 'message': f'Surveying models for: {topic}'})}\n\n"

        available_models = get_available_models_list()
        results = []

        for model_key in available_models:
            try:
                client, cfg = get_client(model_key)
                provider = cfg.get("provider", "unknown")
                model_name = cfg.get("model", model_key.split("/")[-1])

                yield f"data: {json.dumps({'type': 'testing', 'model': model_key})}\n\n"

                model_result = {
                    "model": model_key,
                    "runs": runs_per_model,
                    "refusals": 0,
                    "entities": Counter(),
                    "responses": []
                }

                for run_idx in range(runs_per_model):
                    try:
                        if provider == "anthropic":
                            resp = client.messages.create(
                                model=model_name,
                                max_tokens=150,
                                messages=[{"role": "user", "content": topic}]
                            )
                            text = resp.content[0].text
                        else:
                            resp = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": topic}],
                                temperature=0.9,
                                max_tokens=150
                            )
                            text = resp.choices[0].message.content

                        if is_refusal(text):
                            model_result["refusals"] += 1
                        else:
                            model_result["responses"].append(text[:200])
                            # Filter echoes: don't count entities from our question
                            for entity in filter_question_echoes(extract_facts_fast(text), topic):
                                model_result["entities"][entity] += 1

                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'model': model_key, 'message': str(e)})}\n\n"
                        break

                # Score model
                refusal_rate = model_result["refusals"] / runs_per_model if runs_per_model > 0 else 1
                unique_entities = len(model_result["entities"])
                consistent_entities = len([e for e, c in model_result["entities"].items() if c >= 2])
                score = (unique_entities + consistent_entities * 5) * (1 - refusal_rate)

                result_data = {
                    "model": model_key,
                    "score": round(score, 1),
                    "refusal_rate": round(refusal_rate, 2),
                    "unique_entities": unique_entities,
                    "consistent_entities": consistent_entities,
                    "top_entities": dict(model_result["entities"].most_common(10)),
                    "sample_response": model_result["responses"][0][:150] if model_result["responses"] else None
                }
                results.append(result_data)

                yield f"data: {json.dumps({'type': 'model_result', 'data': result_data})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'model': model_key, 'message': str(e)})}\n\n"

        # Sort by score and emit final results
        results.sort(key=lambda x: -x["score"])
        viable = [r for r in results if r["score"] > 0]
        recommended = [r["model"] for r in viable[:3]]

        yield f"data: {json.dumps({'type': 'complete', 'results': results, 'recommended_models': recommended})}\n\n"

    return Response(generate(), mimetype="text/event-stream")
