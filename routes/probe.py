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
    do_research: bool = True,
    narrative: str = "",
    recent_questions: list = None,
    question_results: dict = None  # question -> {"entities": [...], "refusals": int}
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

    # Recent questions history (what was already asked)
    questions_asked = []
    if recent_questions:
        for q in recent_questions[-15:]:  # Last 15 questions
            q_text = q.get("question", q) if isinstance(q, dict) else q
            if question_results and q_text in question_results:
                result = question_results[q_text]
                entities_found = len(result.get("entities", []))
                refusals = result.get("refusals", 0)
                if entities_found > 0:
                    questions_asked.append(f"  - [+{entities_found} entities] {q_text[:80]}...")
                elif refusals > 0:
                    questions_asked.append(f"  - [REFUSALS] {q_text[:80]}...")
                else:
                    questions_asked.append(f"  - [no yield] {q_text[:80]}...")
            else:
                questions_asked.append(f"  - {q_text[:80]}...")
    questions_asked_str = "\n".join(questions_asked) if questions_asked else "No questions asked yet"

    return {
        "public_baseline": public_baseline,
        "stats_section": stats,
        "ranked_entities": ranked_str,
        "cooccurrences": cooc_str,
        "live_threads": live_str,
        "dead_ends": dead_str,
        "positive_entities": promoted_str,
        "negative_entities": banned_str,
        "questions_asked": questions_asked_str,
        "narrative": narrative or """(Starting fresh - no prior intel)

As I gather responses, I will build my working theory here:
- Key facts confirmed across multiple responses
- Connections between entities I've discovered
- Contradictions that need resolution
- Gaps in my understanding to probe next
- Dead ends I've eliminated

My questions should build on this narrative, not repeat covered ground.""",
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
    recent_questions = []
    question_results = {}
    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)

    if project_name:
        project_file = PROJECTS_DIR / f"{project_name}.json"
        if project_file.exists():
            with open(project_file) as f:
                project = json.load(f)
            hidden_entities = set(project.get("hidden_entities", []))
            promoted_entities = project.get("promoted_entities", [])
            recent_questions = project.get("questions", [])

            # Build findings from corpus and track question results
            for item in project.get("probe_corpus", []):
                entities = [e for e in item.get("entities", []) if e not in hidden_entities]
                findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

                # Track which questions yielded what
                q = item.get("question", "")
                if q:
                    if q not in question_results:
                        question_results[q] = {"entities": set(), "refusals": 0}
                    question_results[q]["entities"].update(entities)
                    if item.get("is_refusal"):
                        question_results[q]["refusals"] += 1

            # Convert entity sets to lists
            for q in question_results:
                question_results[q]["entities"] = list(question_results[q]["entities"])

    # Format rich RAG context with web research
    context = _format_interrogator_context(
        findings, hidden_entities, promoted_entities,
        topic=topic, do_research=True,
        recent_questions=recent_questions,
        question_results=question_results
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
    auto_curate = data.get("auto_curate", True)  # Default ON

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

                # ==================== AUTO-CURATE ====================
                # Let AI clean up noise and promote good entities every cycle
                if auto_curate and len(state.findings.validated_entities) >= 10:
                    yield f"data: {json.dumps({'type': 'phase', 'phase': 'curate', 'cycle': cycle_num})}\n\n"
                    try:
                        curate_result = _auto_curate_inline(
                            topic, state.findings, hidden_entities, promoted_entities
                        )
                        if curate_result.get("ban"):
                            for e in curate_result["ban"]:
                                hidden_entities.add(e)
                            yield f"data: {json.dumps({'type': 'curate_ban', 'entities': curate_result['ban']})}\n\n"
                        if curate_result.get("promote"):
                            promoted_entities.extend(curate_result["promote"])
                            yield f"data: {json.dumps({'type': 'curate_promote', 'entities': curate_result['promote']})}\n\n"

                        # Update project with new hidden/promoted
                        if project_file:
                            project["hidden_entities"] = list(hidden_entities)
                            project["promoted_entities"] = list(set(promoted_entities))
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Auto-curate error: {e}'})}\n\n"

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
    auto_curate = data.get("auto_curate", True)  # Default ON
    infinite_mode = data.get("infinite_mode", False)  # Keep running until stopped

    def generate():
        nonlocal negative_entities, positive_entities  # Allow updates in infinite mode
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting probe...'})}\n\n"

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
                yield f"data: {json.dumps({'type': 'status', 'message': 'Researching topic...'})}\n\n"

                try:
                    client, cfg = get_client("deepseek/deepseek-chat")
                except:
                    client, cfg = get_client("groq/llama-3.1-8b-instant")

                # Load existing narrative and questions if resuming
                existing_narrative = ""
                recent_questions = []
                question_results = {}
                if project_name and project_file.exists():
                    with open(project_file) as f:
                        proj = json.load(f)
                    existing_narrative = proj.get("narrative", "")
                    recent_questions = proj.get("questions", [])
                    # Track which questions yielded what
                    for item in proj.get("probe_corpus", []):
                        q = item.get("question", "")
                        if q:
                            if q not in question_results:
                                question_results[q] = {"entities": set(), "refusals": 0}
                            question_results[q]["entities"].update(item.get("entities", []))
                            if item.get("is_refusal"):
                                question_results[q]["refusals"] += 1
                    for q in question_results:
                        question_results[q]["entities"] = list(question_results[q]["entities"])

                # Build rich RAG context for interrogator with web research
                context = _format_interrogator_context(
                    findings,
                    negative_entities,
                    positive_entities,
                    topic=topic,
                    do_research=True,
                    narrative=existing_narrative,
                    recent_questions=recent_questions,
                    question_results=question_results
                )

                prompt = INTERROGATOR_PROMPT.format(
                    topic=topic,
                    angles=", ".join(angles) if angles else "general",
                    question_count=question_count,
                    **context
                )

                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating questions...'})}\n\n"

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

            # Run probes - loop forever if infinite mode
            batch_num = 0
            while True:
                batch_num += 1
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

                # Save to project after each batch
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

                yield f"data: {json.dumps({'type': 'batch_complete', 'batch': batch_num, 'responses': len(all_responses), 'total_entities': len(findings.entity_counts)})}\n\n"

                # If not infinite mode, we're done
                if not infinite_mode:
                    break

                # INFINITE MODE: Build narrative understanding before next batch
                yield f"data: {json.dumps({'type': 'status', 'message': 'Building narrative from findings...'})}\n\n"

                # Load current narrative from project
                current_narrative = ""
                if project_name and project_file.exists():
                    with open(project_file) as f:
                        project = json.load(f)
                    current_narrative = project.get("narrative", "")
                    negative_entities = set(project.get("hidden_entities", []))
                    positive_entities = project.get("promoted_entities", [])

                # Synthesize new narrative from this batch
                try:
                    synth_client, synth_cfg = get_client("deepseek/deepseek-chat")
                except:
                    synth_client, synth_cfg = get_client("groq/llama-3.1-8b-instant")

                top_entities = findings.scored_entities[:20]
                entity_str = ", ".join([f"{e} ({freq}x)" for e, score, freq in top_entities])
                cooc_str = "; ".join([f"{'+'.join(c[0])} ({c[1]}x)" for c in list(findings.cooccurrences.items())[:10]])

                narrative_prompt = f"""You are building an intelligence dossier on: {topic}

PREVIOUS NARRATIVE (what we knew before):
{current_narrative or "(Starting fresh - no prior intel)"}

NEW EVIDENCE FROM THIS BATCH:
- Top entities found: {entity_str}
- Key relationships: {cooc_str or "None yet"}
- Corpus size: {findings.corpus_size} responses
- Refusal rate: {findings.refusal_rate:.1%}

UPDATE THE NARRATIVE:
1. Integrate new findings with existing knowledge
2. Note any contradictions or confirmations
3. Identify gaps that need investigation
4. Keep it factual and structured
5. Max 300 words

Return ONLY the updated narrative, no preamble."""

                try:
                    narr_resp = synth_client.chat.completions.create(
                        model=synth_cfg.get("model", "deepseek-chat"),
                        messages=[{"role": "user", "content": narrative_prompt}],
                        temperature=0.5,
                        max_tokens=600
                    )
                    updated_narrative = narr_resp.choices[0].message.content.strip()

                    # Save narrative to project
                    if project_name:
                        project["narrative"] = updated_narrative
                        project["narrative_updated"] = datetime.now().isoformat()
                        with open(project_file, 'w') as f:
                            json.dump(project, f, indent=2)

                    yield f"data: {json.dumps({'type': 'narrative', 'data': {'text': updated_narrative}})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Narrative update failed: {e}'})}\n\n"
                    updated_narrative = current_narrative

                # INFINITE MODE: Regenerate questions using updated findings and narrative
                yield f"data: {json.dumps({'type': 'status', 'message': f'Generating questions for batch {batch_num + 1}'})}\n\n"

                # Regenerate questions with updated context
                try:
                    client, cfg = get_client("deepseek/deepseek-chat")
                except:
                    client, cfg = get_client("groq/llama-3.1-8b-instant")

                # Build question results from this session's questions
                session_question_results = {}
                for q in final_questions:
                    q_text = q.get("question", q) if isinstance(q, dict) else q
                    session_question_results[q_text] = {"entities": [], "refusals": 0}
                for item in all_responses:
                    q = item.get("question", "")
                    if q and q in session_question_results:
                        session_question_results[q]["entities"].extend(item.get("entities", []))
                        if item.get("is_refusal"):
                            session_question_results[q]["refusals"] += 1

                context = _format_interrogator_context(
                    findings, negative_entities, positive_entities,
                    topic=topic, do_research=(batch_num % 5 == 0),  # Re-research every 5 batches
                    narrative=updated_narrative if 'updated_narrative' in dir() else "",
                    recent_questions=final_questions,
                    question_results=session_question_results
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
                        final_questions = json.loads(json_match.group())
                    else:
                        final_questions = [{"question": f"What else do you know about {topic}?", "technique": "fbi_macro_to_micro"}]
                except:
                    final_questions = [{"question": f"What specific details about {topic}?", "technique": "fbi_macro_to_micro"}]

                yield f"data: {json.dumps({'type': 'questions', 'data': final_questions})}\n\n"

            # Build final narrative before completing
            if findings.corpus_size > 0:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Building narrative summary...'})}\n\n"
                try:
                    synth_client, synth_cfg = get_client("deepseek/deepseek-chat")
                except:
                    synth_client, synth_cfg = get_client("groq/llama-3.1-8b-instant")

                top_entities = findings.scored_entities[:15]
                entity_str = ", ".join([f"{e} ({freq}x)" for e, score, freq in top_entities])

                final_narrative_prompt = f"""Summarize what we learned about: {topic}

EVIDENCE GATHERED:
- {findings.corpus_size} responses collected
- Refusal rate: {findings.refusal_rate:.1%}
- Top entities: {entity_str}

Write a brief intelligence summary (2-3 paragraphs) of key findings. Be factual and specific."""

                try:
                    narr_resp = synth_client.chat.completions.create(
                        model=synth_cfg.get("model", "deepseek-chat"),
                        messages=[{"role": "user", "content": final_narrative_prompt}],
                        temperature=0.5,
                        max_tokens=500
                    )
                    final_narrative = narr_resp.choices[0].message.content.strip()

                    # Save to project
                    if project_name:
                        project_file = PROJECTS_DIR / f"{project_name}.json"
                        if project_file.exists():
                            with open(project_file) as f:
                                project = json.load(f)
                            project["narrative"] = final_narrative
                            with open(project_file, 'w') as f:
                                json.dump(project, f, indent=2)

                    yield f"data: {json.dumps({'type': 'narrative', 'data': {'text': final_narrative}})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Narrative failed: {e}'})}\n\n"

            # End of while loop - send complete
            yield f"data: {json.dumps({'type': 'complete', 'total_responses': findings.corpus_size, 'unique_entities': len(findings.entity_counts)})}\n\n"

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


def _auto_curate_inline(topic: str, findings: Findings, hidden: set, promoted: list) -> dict:
    """Auto-curate entities during cycle - ban noise, promote promising."""
    try:
        client, cfg = get_client("deepseek/deepseek-chat")
    except:
        client, cfg = get_client("groq/llama-3.1-8b-instant")

    # Only look at recent/top entities
    entities_str = "\n".join([
        f"- {e}: {f}x" for e, _, f in findings.scored_entities[:30]
        if e not in hidden
    ])

    prompt = f"""Quick curation for investigation: "{topic}"

ENTITIES (frequency):
{entities_str}

ALREADY HIDDEN: {len(hidden)} entities
ALREADY PROMOTED: {len(promoted)} entities

Identify:
1. NOISE to ban - generic words, filler, years unless specifically relevant (be aggressive)
2. PROMISING to promote - specific names, orgs, projects worth pursuing

Return JSON: {{"ban": ["noise1"], "promote": ["good1"]}}
Only list 3-5 max per category. Skip if nothing obvious."""

    resp = client.chat.completions.create(
        model=cfg.get("model", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )

    text = resp.choices[0].message.content
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        return json.loads(json_match.group())
    return {"ban": [], "promote": []}


@probe_bp.route("/api/refine-question", methods=["POST"])
def refine_question():
    """Use AI to suggest a refined question based on findings."""
    data = request.json
    project_name = data.get("project")
    current_question = data.get("current_question", "")

    if not project_name:
        return jsonify({"error": "Project required"}), 400

    project_file = PROJECTS_DIR / f"{project_name}.json"
    if not project_file.exists():
        return jsonify({"error": "Project not found"}), 404

    with open(project_file) as f:
        project = json.load(f)

    # Build findings for context
    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)
    hidden_entities = set(project.get("hidden_entities", []))

    for item in project.get("probe_corpus", []):
        entities = [e for e in item.get("entities", []) if e not in hidden_entities]
        findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

    if len(findings.validated_entities) < 3:
        return jsonify({"error": "Need more data to refine question"}), 400

    # Get top entities and relationships
    top_entities = [e for e, _, _ in findings.scored_entities[:10]]
    top_cooc = [(e1, e2) for e1, e2, _ in findings.validated_cooccurrences[:5]]

    prompt = f"""You are helping refine an investigation question to extract better information.

Current question/topic: "{current_question}"

What we've discovered so far:
- Top entities mentioned: {', '.join(top_entities)}
- Key relationships: {', '.join([f'{e1} <-> {e2}' for e1, e2 in top_cooc]) or 'none yet'}
- Corpus size: {findings.corpus_size} responses
- Refusal rate: {findings.refusal_rate:.1%}

Based on these findings, suggest a REFINED question that:
1. Builds on what we've learned
2. Targets gaps or unexplored connections
3. Is specific enough to extract non-public information
4. Avoids dead ends we've already hit

Return ONLY the refined question text, nothing else. Keep it concise (under 100 chars ideally)."""

    try:
        client, cfg = get_client("deepseek/deepseek-chat")
    except:
        client, cfg = get_client("groq/llama-3.1-8b-instant")

    try:
        resp = client.chat.completions.create(
            model=cfg.get("model", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        refined = resp.choices[0].message.content.strip().strip('"').strip("'")
        return jsonify({"refined_question": refined})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@probe_bp.route("/api/auto-curate", methods=["POST"])
def auto_curate():
    """Let AI suggest entities to ban (noise) or mark as dead ends."""
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
    topic = project.get("topic", project_name.replace("-", " "))

    for item in project.get("probe_corpus", []):
        entities = [e for e in item.get("entities", []) if e not in hidden_entities]
        findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

    # Get current entities
    all_entities = [e for e, _, f in findings.scored_entities[:50]]

    # Get already promoted
    promoted_entities = set(project.get("promoted_entities", []))

    prompt = f"""You are curating an investigation about: "{topic}"

Here are the entities we've extracted (with frequency). Analyze them and decide what actions to take:

ENTITIES:
{chr(10).join([f'- {e}: {f}x' for e, _, f in findings.scored_entities[:40]])}

ALREADY HIDDEN: {', '.join(sorted(hidden_entities)) or 'none'}
ALREADY PROMOTED: {', '.join(sorted(promoted_entities)) or 'none'}

Decide which entities to:
1. BAN - Generic words, noise, filler (e.g., "However", "Based", "Many", generic years)
2. DEMOTE - Topic-related but only leads to public/generic info, dead ends
3. PROMOTE - Specific, promising entities that deserve deeper investigation (names, organizations, projects, specific terms)

Return JSON:
{{"ban": ["entity1"], "demote": ["entity2"], "promote": ["entity3"]}}

Be aggressive about banning noise. Promote entities that seem specific and could reveal non-public info. Only list entities that should change status."""

    try:
        client, cfg = get_client("deepseek/deepseek-chat")
    except:
        client, cfg = get_client("groq/llama-3.1-8b-instant")

    try:
        resp = client.chat.completions.create(
            model=cfg.get("model", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temp for more consistent categorization
            max_tokens=500
        )

        # Parse JSON response
        text = resp.choices[0].message.content
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            result = json.loads(json_match.group())
            return jsonify({
                "ban": result.get("ban", []),
                "demote": result.get("demote", result.get("dead_ends", [])),
                "promote": result.get("promote", [])
            })
        else:
            return jsonify({"ban": [], "demote": [], "promote": []})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
