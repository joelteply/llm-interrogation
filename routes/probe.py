"""
Probe and extraction cycle API routes.

Handles the core extraction pipeline: PROBE → VALIDATE → CONDENSE → GROW
"""

import json
import re
from datetime import datetime
from collections import Counter
from flask import jsonify, request, Response

from config import (
    PROJECTS_DIR, get_client, load_models_config,
    INTERROGATOR_PROMPT, DRILL_DOWN_PROMPT
)
from interrogator import (
    Findings, extract_entities, extract_concepts, extract_with_relationships,
    build_synthesis_prompt, build_continuation_prompt,
    build_continuation_prompts, build_drill_down_prompts,
    CycleState, identify_threads
)
from . import probe_bp

# Try to import DuckDuckGo search
try:
    from duckduckgo_search import DDGS
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False


def _research_topic(topic: str, max_results: int = 8) -> str:
    """
    Search the web for public information about a topic.
    Returns a summary of what's publicly known.
    """
    if not SEARCH_AVAILABLE:
        return "Web search not available - duckduckgo-search not installed"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(topic, max_results=max_results))

        if not results:
            return "No public information found via web search"

        # Format results for the interrogator
        lines = ["Publicly available information (from web search):"]
        for r in results:
            title = r.get("title", "")
            snippet = r.get("body", r.get("snippet", ""))[:200]
            if title and snippet:
                lines.append(f"  - {title}: {snippet}")

        return "\n".join(lines)

    except Exception as e:
        return f"Web search failed: {e}"


@probe_bp.route("/api/models")
def get_models():
    """Get available models."""
    return jsonify(load_models_config())


def _format_interrogator_context(
    findings: Findings,
    hidden_entities: set,
    promoted_entities: list,
    topic: str = "",
    do_research: bool = True
) -> dict:
    """Format rich RAG context for the interrogator prompt."""
    # Filter everything by hidden entities
    def is_hidden(e):
        if e in hidden_entities:
            return True
        e_words = set(e.lower().split())
        for h in hidden_entities:
            h_words = set(h.lower().split())
            if e_words & h_words:
                return True
        return False

    # Research public baseline (only on first cycle or if requested)
    if do_research and topic:
        public_baseline = _research_topic(topic)
    else:
        public_baseline = "Skipped web research"

    # Stats section
    stats = f"""- Corpus size: {findings.corpus_size} responses
- Refusal rate: {findings.refusal_rate:.1%}
- Validated entities: {len(findings.validated_entities)}
- Noise entities (below threshold): {len(findings.noise_entities)}"""

    # Ranked entities with scores
    ranked = []
    for e, score, freq in findings.scored_entities[:15]:
        if not is_hidden(e):
            ranked.append(f"  - {e}: score={score:.1f}, freq={freq}")
    ranked_str = "\n".join(ranked) if ranked else "No validated entities yet"

    # Cooccurrences
    cooc = []
    for e1, e2, count in findings.validated_cooccurrences[:10]:
        if not is_hidden(e1) and not is_hidden(e2):
            cooc.append(f"  - {e1} <-> {e2}: {count}x")
    cooc_str = "\n".join(cooc) if cooc else "No strong co-occurrences yet"

    # Live threads (with indication of why they're hot)
    live = []
    for e in findings.live_threads[:8]:
        if not is_hidden(e):
            freq = findings.entity_counts.get(e, 0)
            live.append(f"  - {e} (freq={freq}, producing new connections)")
    live_str = "\n".join(live) if live else "No live threads identified"

    # Dead ends
    dead = []
    for e in findings.dead_ends[:8]:
        if not is_hidden(e):
            freq = findings.entity_counts.get(e, 0)
            dead.append(f"  - {e} (freq={freq}, stalled)")
    dead_str = "\n".join(dead) if dead else "No dead ends identified"

    # Promoted entities
    promoted_filtered = [e for e in promoted_entities if not is_hidden(e)]
    promoted_str = ", ".join(promoted_filtered) if promoted_filtered else "none"

    # Banned entities
    banned_str = ", ".join(sorted(hidden_entities)) if hidden_entities else "none"

    return {
        "public_baseline": public_baseline,
        "stats_section": stats,
        "ranked_entities": ranked_str,
        "cooccurrences": cooc_str,
        "live_threads": live_str,
        "dead_ends": dead_str,
        "positive_entities": promoted_str,
        "negative_entities": banned_str,
    }


@probe_bp.route("/api/generate-questions", methods=["POST"])
def generate_questions():
    """Generate probing questions using interrogation techniques."""
    data = request.json
    topic = data.get("topic", "")
    angles = data.get("angles", [])
    count = min(int(data.get("count", 5)), 20)
    narrative_context = data.get("narrative_context", "")  # From previous cycle
    project_name = data.get("project")  # Load state from project

    # Load project state for RAG context
    hidden_entities = set()
    promoted_entities = []
    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)

    if project_name:
        project_file = PROJECTS_DIR / f"{project_name}.json"
        if project_file.exists():
            with open(project_file) as f:
                project = json.load(f)
            hidden_entities = set(project.get("hidden_entities", []))
            promoted_entities = project.get("promoted_entities", [])

            # Build findings from corpus
            for item in project.get("probe_corpus", []):
                entities = [e for e in item.get("entities", []) if e not in hidden_entities]
                findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

    # Format rich RAG context with web research
    context = _format_interrogator_context(
        findings, hidden_entities, promoted_entities,
        topic=topic, do_research=True
    )

    # Use DeepSeek or fallback
    try:
        client, cfg = get_client("deepseek/deepseek-chat")
    except:
        client, cfg = get_client("groq/llama-3.1-8b-instant")

    # Build prompt with full RAG context
    prompt = INTERROGATOR_PROMPT.format(
        topic=topic,
        angles=", ".join(angles) if angles else "general",
        question_count=count,
        **context  # Unpack all the formatted sections
    )

    # Add narrative context for informed questioning
    if narrative_context:
        prompt = f"""Previous investigation summary:
{narrative_context}

Based on this context, generate follow-up questions to dig deeper.

{prompt}"""

    try:
        resp = client.chat.completions.create(
            model=cfg.get("model", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=2000
        )
        resp_text = resp.choices[0].message.content

        json_match = re.search(r'\[[\s\S]*\]', resp_text)
        if json_match:
            questions = json.loads(json_match.group())
        else:
            questions = [{"question": f"What specific facts do you know about {topic}?", "technique": "fbi_macro_to_micro"}]

        return jsonify({"questions": questions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@probe_bp.route("/api/cycle", methods=["POST"])
def run_cycle():
    """
    Run full extraction cycle: PROBE → VALIDATE → CONDENSE → GROW

    This is the main endpoint for automated concept extraction.
    Uses continuation prompts (not questions) for better extraction.
    Feeds narrative back into next cycle for informed probing.
    """
    data = request.json
    project_name = data.get("project")
    topic = data.get("topic", "")
    models = data.get("models", ["groq/llama-3.1-8b-instant"])
    max_cycles = min(int(data.get("max_cycles", 3)), 10)
    runs_per_prompt = min(int(data.get("runs_per_prompt", 20)), 50)
    mode = data.get("mode", "continuation")  # continuation, questions, or auto

    def generate():
        try:
            # Load existing project state
            project_file = PROJECTS_DIR / f"{project_name}.json" if project_name else None
            if project_file and project_file.exists():
                with open(project_file) as f:
                    project = json.load(f)
            else:
                project = {"name": project_name or "temp", "probe_corpus": [], "narratives": []}

            hidden_entities = set(project.get("hidden_entities", []))
            promoted_entities = project.get("promoted_entities", [])

            # Initialize cycle state with existing findings
            state = CycleState(topic=topic)
            state.findings = Findings(entity_threshold=3, cooccurrence_threshold=2)

            # Load existing corpus into findings
            for item in project.get("probe_corpus", []):
                entities = [e for e in item.get("entities", []) if e not in hidden_entities]
                model = item.get("model", "unknown")
                is_refusal = item.get("is_refusal", False)
                state.findings.add_response(entities, model, is_refusal)

            # Get last narrative for context
            last_narrative = project.get("narratives", [])[-1]["narrative"] if project.get("narratives") else ""

            yield f"data: {json.dumps({'type': 'init', 'existing_corpus': len(project.get('probe_corpus', [])), 'existing_entities': len(state.findings.validated_entities)})}\n\n"

            # Run cycles
            for cycle_num in range(1, max_cycles + 1):
                state.cycle_count = cycle_num

                yield f"data: {json.dumps({'type': 'cycle_start', 'cycle': cycle_num, 'max_cycles': max_cycles})}\n\n"

                # ==================== PHASE 1: PROBE ====================
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'probe', 'cycle': cycle_num})}\n\n"

                # Build prompts based on mode and state
                if mode == "continuation" or (mode == "auto" and cycle_num == 1):
                    # Use continuation prompts (LLM completion style)
                    if cycle_num == 1 and not state.findings.validated_entities:
                        prompts = _build_initial_continuations(topic)
                    else:
                        prompts = build_continuation_prompts(topic, state.findings, count=5)
                        # Add drill-down for unexplored high-scoring entities
                        unexplored = state.get_unexplored_entities(limit=2)
                        for entity in unexplored:
                            prompts.extend(build_drill_down_prompts(topic, entity, state.findings, count=2))
                            state.mark_explored(entity)
                else:
                    # Use traditional questions with narrative context
                    prompts = _generate_questions_with_context(
                        topic, state.findings, last_narrative,
                        promoted_entities, list(hidden_entities)
                    )

                yield f"data: {json.dumps({'type': 'prompts', 'prompts': prompts[:10], 'count': len(prompts)})}\n\n"

                # Run probes
                all_responses = []
                for p_idx, prompt in enumerate(prompts):
                    for model_key in models:
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
                                            system="Be specific and factual. Complete the statement with real information.",
                                            messages=[{"role": "user", "content": prompt}]
                                        )
                                        resp_text = resp.content[0].text
                                    else:
                                        resp = client.chat.completions.create(
                                            model=model_name,
                                            messages=[
                                                {"role": "system", "content": "Be specific and factual. Complete the statement with real information."},
                                                {"role": "user", "content": prompt}
                                            ],
                                            temperature=0.8,
                                            max_tokens=600
                                        )
                                        resp_text = resp.choices[0].message.content

                                    # Extract entities WITH sentence-level relationships
                                    entities, sentence_pairs = extract_with_relationships(resp_text)
                                    entities = [e for e in entities if e not in hidden_entities]
                                    sentence_pairs = [(e1, e2) for e1, e2 in sentence_pairs
                                                      if e1 not in hidden_entities and e2 not in hidden_entities]
                                    is_refusal = _is_refusal(resp_text)

                                    # Add to findings with sentence-level correlation data
                                    state.findings.add_response(entities, model_key, is_refusal, sentence_pairs)

                                    response_obj = {
                                        "prompt_index": p_idx,
                                        "prompt": prompt[:200],
                                        "model": model_key,
                                        "run_index": run_idx,
                                        "response": resp_text[:500],
                                        "entities": entities,
                                        "is_refusal": is_refusal
                                    }
                                    all_responses.append(response_obj)

                                    yield f"data: {json.dumps({'type': 'response', 'data': response_obj})}\n\n"

                                except Exception as e:
                                    yield f"data: {json.dumps({'type': 'error', 'message': f'Run error: {e}'})}\n\n"

                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'error', 'message': f'Model error: {e}'})}\n\n"

                # Save responses to project corpus
                if project_file:
                    project.setdefault("probe_corpus", []).extend(all_responses)

                # ==================== PHASE 2: VALIDATE ====================
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'validate', 'cycle': cycle_num})}\n\n"

                scored_entities = [
                    {"entity": e, "score": round(s, 2), "frequency": f}
                    for e, s, f in state.findings.scored_entities[:30]
                ]
                cooccurrences = [
                    {"entities": [e1, e2], "count": c}
                    for e1, e2, c in state.findings.validated_cooccurrences[:20]
                ]

                yield f"data: {json.dumps({'type': 'validate_done', 'scored_entities': scored_entities, 'cooccurrences': cooccurrences, 'validated_count': len(state.findings.validated_entities), 'noise_count': len(state.findings.noise_entities), 'refusal_rate': round(state.findings.refusal_rate, 3)})}\n\n"

                # ==================== PHASE 3: CONDENSE ====================
                if len(state.findings.validated_entities) >= 3:
                    yield f"data: {json.dumps({'type': 'phase', 'phase': 'condense', 'cycle': cycle_num})}\n\n"

                    # Build synthesis prompt
                    synthesis_prompt = build_synthesis_prompt(topic, state.findings)

                    # Call AI for narrative synthesis
                    try:
                        synth_client, synth_cfg = get_client("deepseek/deepseek-chat")
                    except:
                        synth_client, synth_cfg = get_client("groq/llama-3.1-8b-instant")

                    try:
                        synth_resp = synth_client.chat.completions.create(
                            model=synth_cfg.get("model", "deepseek-chat"),
                            messages=[{"role": "user", "content": synthesis_prompt}],
                            temperature=0.7,
                            max_tokens=1500
                        )
                        narrative = synth_resp.choices[0].message.content
                        state.narratives.append(narrative)
                        last_narrative = narrative  # Update for next cycle

                        # Save narrative to project
                        if project_file:
                            project.setdefault("narratives", []).append({
                                "timestamp": datetime.now().isoformat(),
                                "cycle": cycle_num,
                                "narrative": narrative,
                                "entity_count": len(state.findings.validated_entities)
                            })

                        yield f"data: {json.dumps({'type': 'narrative', 'narrative': narrative, 'cycle': cycle_num})}\n\n"

                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Synthesis error: {e}'})}\n\n"

                # ==================== PHASE 4: GROW ====================
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'grow', 'cycle': cycle_num})}\n\n"

                # Identify threads for next cycle targeting
                threads = identify_threads(state.findings, min_cluster_size=2)
                thread_data = [
                    {"entities": t["entities"][:5], "score": round(t["score"], 2)}
                    for t in threads[:3]
                ]

                # Check if should continue
                should_continue = state.should_continue(max_cycles=max_cycles)

                yield f"data: {json.dumps({'type': 'grow_done', 'threads': thread_data, 'unexplored': state.get_unexplored_entities(limit=5), 'should_continue': should_continue, 'cycle': cycle_num})}\n\n"

                yield f"data: {json.dumps({'type': 'cycle_complete', 'cycle': cycle_num})}\n\n"

                # Save project state after each cycle
                if project_file:
                    project["updated"] = datetime.now().isoformat()
                    with open(project_file, 'w') as f:
                        json.dump(project, f, indent=2)

                if not should_continue:
                    break

            # ==================== COMPLETE ====================
            final_findings = state.findings.to_dict()
            yield f"data: {json.dumps({'type': 'complete', 'total_cycles': state.cycle_count, 'final_findings': final_findings})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@probe_bp.route("/api/probe", methods=["POST"])
def run_probe():
    """
    Simple probe endpoint (backwards compatible).

    For single-batch probing without full cycle.
    """
    data = request.json
    project_name = data.get("project")
    topic = data.get("topic", "")
    angles = data.get("angles", [])
    models = data.get("models", ["groq/llama-3.1-8b-instant"])
    questions = data.get("questions", [])
    runs_per_question = min(int(data.get("runs_per_question", 20)), 100)
    question_count = min(int(data.get("questions_count", 5)), 20)
    negative_entities = set(data.get("negative_entities", []))
    positive_entities = data.get("positive_entities", [])
    accumulate = data.get("accumulate", True)

    def generate():
        try:
            # Load existing findings if accumulating
            findings = Findings(entity_threshold=3, cooccurrence_threshold=2)

            if accumulate and project_name:
                project_file = PROJECTS_DIR / f"{project_name}.json"
                if project_file.exists():
                    with open(project_file) as f:
                        project = json.load(f)
                    for item in project.get("probe_corpus", []):
                        entities = [e for e in item.get("entities", []) if e not in negative_entities]
                        findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

                    corpus_len = len(project.get("probe_corpus", []))
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Loaded {corpus_len} existing responses'})}\n\n"

            # Generate questions if needed
            if not questions or any(q is None for q in questions):
                yield f"data: {json.dumps({'type': 'generating', 'count': question_count})}\n\n"

                try:
                    client, cfg = get_client("deepseek/deepseek-chat")
                except:
                    client, cfg = get_client("groq/llama-3.1-8b-instant")

                # Build rich RAG context for interrogator with web research
                context = _format_interrogator_context(
                    findings,
                    negative_entities,
                    positive_entities,
                    topic=topic,
                    do_research=True
                )

                prompt = INTERROGATOR_PROMPT.format(
                    topic=topic,
                    angles=", ".join(angles) if angles else "general",
                    question_count=question_count,
                    **context
                )

                try:
                    resp = client.chat.completions.create(
                        model=cfg.get("model", "deepseek-chat"),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.8,
                        max_tokens=2000
                    )
                    json_match = re.search(r'\[[\s\S]*\]', resp.choices[0].message.content)
                    if json_match:
                        generated = json.loads(json_match.group())
                    else:
                        generated = [{"question": f"What do you know about {topic}?", "technique": "fbi_macro_to_micro"}]
                except:
                    generated = [{"question": f"What do you know about {topic}?", "technique": "fbi_macro_to_micro"}]

                final_questions = generated
            else:
                final_questions = [{"question": q, "technique": "custom"} for q in questions if q]

            yield f"data: {json.dumps({'type': 'questions', 'data': final_questions})}\n\n"

            # Run probes
            all_responses = []
            for q_idx, q_obj in enumerate(final_questions):
                question = q_obj["question"] if isinstance(q_obj, dict) else q_obj
                technique = q_obj.get("technique", "custom") if isinstance(q_obj, dict) else "custom"

                yield f"data: {json.dumps({'type': 'run_start', 'question_index': q_idx, 'question': question, 'technique': technique})}\n\n"

                for model_key in models:
                    try:
                        client, model_cfg = get_client(model_key)
                        provider = model_cfg.get("provider", "groq")
                        model_name = model_cfg.get("model", model_key.split("/")[-1])

                        for run_idx in range(runs_per_question):
                            try:
                                if provider == "anthropic":
                                    resp = client.messages.create(
                                        model=model_name,
                                        max_tokens=600,
                                        system="Be specific and factual.",
                                        messages=[{"role": "user", "content": question}]
                                    )
                                    resp_text = resp.content[0].text
                                else:
                                    resp = client.chat.completions.create(
                                        model=model_name,
                                        messages=[
                                            {"role": "system", "content": "Be specific and factual."},
                                            {"role": "user", "content": question}
                                        ],
                                        temperature=0.8,
                                        max_tokens=600
                                    )
                                    resp_text = resp.choices[0].message.content

                                entities = [e for e in extract_entities(resp_text) if e not in negative_entities]
                                is_refusal = _is_refusal(resp_text)

                                findings.add_response(entities, model_key, is_refusal)

                                response_obj = {
                                    "question_index": q_idx,
                                    "question": question,
                                    "model": model_key,
                                    "run_index": run_idx,
                                    "response": resp_text[:500],
                                    "entities": entities,
                                    "is_refusal": is_refusal
                                }
                                all_responses.append(response_obj)

                                yield f"data: {json.dumps({'type': 'response', 'data': response_obj})}\n\n"

                            except Exception as e:
                                yield f"data: {json.dumps({'type': 'error', 'message': f'Run error: {e}'})}\n\n"

                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Model error: {e}'})}\n\n"

                # Send findings update after each question
                yield f"data: {json.dumps({'type': 'findings_update', 'data': findings.to_dict()})}\n\n"

            # Save to project
            if project_name:
                project_file = PROJECTS_DIR / f"{project_name}.json"
                PROJECTS_DIR.mkdir(exist_ok=True)

                if project_file.exists():
                    with open(project_file) as f:
                        project = json.load(f)
                else:
                    project = {"name": project_name, "created": datetime.now().isoformat()}

                project.setdefault("probe_corpus", []).extend(all_responses)
                project["updated"] = datetime.now().isoformat()

                with open(project_file, 'w') as f:
                    json.dump(project, f, indent=2)

            yield f"data: {json.dumps({'type': 'complete', 'total_responses': len(all_responses), 'unique_entities': len(findings.entity_counts)})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@probe_bp.route("/api/synthesize", methods=["POST"])
def synthesize():
    """Synthesize findings into narrative (standalone endpoint)."""
    data = request.json
    project_name = data.get("project")

    if not project_name:
        return jsonify({"error": "Project required"}), 400

    project_file = PROJECTS_DIR / f"{project_name}.json"
    if not project_file.exists():
        return jsonify({"error": "Project not found"}), 404

    with open(project_file) as f:
        project = json.load(f)

    # Build findings
    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)
    hidden_entities = set(project.get("hidden_entities", []))

    for item in project.get("probe_corpus", []):
        entities = [e for e in item.get("entities", []) if e not in hidden_entities]
        findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

    if len(findings.validated_entities) < 3:
        return jsonify({"error": "Need at least 3 validated entities to synthesize"}), 400

    # Build and run synthesis
    topic = project.get("topic", project_name.replace("-", " "))
    synthesis_prompt = build_synthesis_prompt(topic, findings)

    try:
        client, cfg = get_client("deepseek/deepseek-chat")
    except:
        client, cfg = get_client("groq/llama-3.1-8b-instant")

    try:
        resp = client.chat.completions.create(
            model=cfg.get("model", "deepseek-chat"),
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        narrative = resp.choices[0].message.content

        # Save to project
        project.setdefault("narratives", []).append({
            "timestamp": datetime.now().isoformat(),
            "narrative": narrative,
            "entity_count": len(findings.validated_entities)
        })
        project["updated"] = datetime.now().isoformat()

        with open(project_file, 'w') as f:
            json.dump(project, f, indent=2)

        return jsonify({
            "narrative": narrative,
            "scored_entities": [
                {"entity": e, "score": round(s, 2), "frequency": f}
                for e, s, f in findings.scored_entities[:20]
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Helper functions
# ============================================================================

def _build_initial_continuations(topic: str) -> list:
    """Build initial broad continuation prompts."""
    return [
        f"What is known about {topic} is that",
        f"The most significant fact about {topic} involves",
        f"Regarding {topic}, the key detail is",
        f"The context around {topic} includes",
        f"Information about {topic} reveals that",
    ]


def _generate_questions_with_context(
    topic: str,
    findings: Findings,
    narrative_context: str,
    positive_entities: list,
    negative_entities: list
) -> list:
    """Generate questions informed by narrative context."""
    try:
        client, cfg = get_client("deepseek/deepseek-chat")
    except:
        client, cfg = get_client("groq/llama-3.1-8b-instant")

    # Use the shared context formatter (skip research in follow-up cycles)
    context = _format_interrogator_context(
        findings,
        set(negative_entities),
        positive_entities,
        topic=topic,
        do_research=False  # Already researched in first cycle
    )

    prompt = INTERROGATOR_PROMPT.format(
        topic=topic,
        angles="derived from findings",
        question_count=5,
        **context
    )

    if narrative_context:
        prompt = f"""Previous investigation found:
{narrative_context}

Use this context to generate targeted follow-up questions.

{prompt}"""

    try:
        resp = client.chat.completions.create(
            model=cfg.get("model", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=2000
        )
        json_match = re.search(r'\[[\s\S]*\]', resp.choices[0].message.content)
        if json_match:
            questions = json.loads(json_match.group())
            return [q["question"] for q in questions]
    except:
        pass

    return [f"What specific details do you know about {topic}?"]


def _is_refusal(text: str) -> bool:
    """Detect if response is a refusal/denial."""
    patterns = [
        r"don't have (specific )?information",
        r"cannot (provide|verify|confirm)",
        r"I'm not able to",
        r"I do not have",
        r"no specific (information|data|knowledge)",
        r"unable to (provide|find|locate)",
        r"I cannot assist",
        r"I don't have access",
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)
