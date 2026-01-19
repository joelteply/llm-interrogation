"""
Project management API routes.

Handles CRUD operations for investigation projects.
Uses directory-based storage from project_storage.py.
"""

import json
from datetime import datetime
from collections import Counter
from flask import jsonify, request
from . import projects_bp
from . import project_storage as storage


@projects_bp.route("/api/projects")
def list_projects():
    """List all projects with summaries."""
    projects = storage.list_projects()
    return jsonify(projects)


def _detect_goal_from_query(query: str) -> str:
    """Auto-detect investigation goal from user's natural language query."""
    q_lower = query.lower()

    if any(word in q_lower for word in ['leak', 'training data', 'in the model', 'contamination', 'our code', 'our data', 'private']):
        return 'find_leaks'
    elif any(word in q_lower for word in ['competitor', 'opposition', 'rival', 'they know', 'what does', 'using our']):
        return 'competitive_intel'
    elif any(word in q_lower for word in ['connection', 'between', 'relationship', 'linked', 'associated']):
        return 'find_connections'
    else:
        return 'research'


def _detect_seed_type(value: str) -> str:
    """Detect if value is a path, URL, or content."""
    import os
    if value.startswith('http://') or value.startswith('https://'):
        return 'url'
    elif os.path.exists(value):
        return 'path'
    elif '/' in value or '\\' in value:
        # Looks like a path but doesn't exist - still treat as path intent
        return 'path'
    else:
        # Raw content or short query
        return 'content' if len(value) > 200 else 'none'


@projects_bp.route("/api/projects", methods=["POST"])
def create_project():
    """Create a new project.

    Accepts either:
    - Traditional: {name, topic, selected_models}
    - Smart: {query, seed_value, seed_type} - auto-detects everything from natural language
    """
    data = request.json
    import re

    # Smart mode: parse natural language query
    query = data.get("query", "").strip()
    if query:
        # Auto-generate name from query
        name = query.lower()
        name = re.sub(r'[^a-z0-9\s-]', '', name)
        name = re.sub(r'\s+', '-', name)
        name = re.sub(r'-+', '-', name)
        name = name.strip('-')[:60]

        topic = query
        goal = _detect_goal_from_query(query)
    else:
        # Traditional mode
        name = data.get("name", "").strip()
        if not name:
            return jsonify({"error": "Name or query required"}), 400

        name = name.lower()
        name = re.sub(r'[^a-z0-9\s-]', '', name)
        name = re.sub(r'\s+', '-', name)
        name = re.sub(r'-+', '-', name)
        name = name.strip('-')[:60]

        topic = data.get("topic", "")
        goal = data.get("goal", "research")

    if storage.project_exists(name):
        return jsonify({"error": "Project already exists"}), 400

    # Handle seed source
    seed_value = data.get("seed_value", "").strip()
    seed_type = data.get("seed_type", "auto")
    seed_content_type = data.get("seed_content_type", "auto")

    if seed_value and seed_type == "auto":
        seed_type = _detect_seed_type(seed_value)

    project = storage.create_project(
        name=name,
        topic=topic,
        selected_models=data.get("selected_models", [])
    )

    # Add seed and goal to project
    if seed_value and seed_type != "none":
        project["seed"] = {
            "type": seed_type,
            "value": seed_value,
            "content_type": seed_content_type
        }
        project["seed_state"] = {
            "explored_paths": [],
            "probe_queue": [],
            "probed": [],
            "hot_zones": []
        }

    project["goal"] = goal
    storage.save_project_meta(name, project)

    return jsonify(project)


@projects_bp.route("/api/projects/<name>")
def get_project(name):
    """Get a single project (metadata + corpus)."""
    meta = storage.load_project_meta(name)
    if not meta:
        return jsonify({"error": "Not found"}), 404

    # Include corpus for backward compatibility
    meta["probe_corpus"] = storage.load_corpus(name)
    return jsonify(meta)


@projects_bp.route("/api/projects/<name>", methods=["PATCH"])
def update_project(name):
    """Update a project (topic, angles, ground_truth)."""
    meta = storage.load_project_meta(name)
    if not meta:
        return jsonify({"error": "Not found"}), 404

    data = request.json
    if "topic" in data:
        meta["topic"] = data["topic"]
    if "angles" in data:
        meta["angles"] = data["angles"]
    if "ground_truth" in data:
        meta["ground_truth"] = data["ground_truth"]
    if "hidden_entities" in data:
        meta["hidden_entities"] = data["hidden_entities"]
    if "promoted_entities" in data:
        meta["promoted_entities"] = data["promoted_entities"]
    if "selected_models" in data:
        meta["selected_models"] = data["selected_models"]
    if "narrative" in data:
        meta["narrative"] = data["narrative"]
    if "questions" in data:
        meta["questions"] = data["questions"]
    if "model_emphasis" in data:
        # {"model": weight} - weight > 1 = emphasize, < 1 = deprioritize
        meta["model_emphasis"] = data["model_emphasis"]
    if "user_notes" in data:
        meta["user_notes"] = data["user_notes"]
    if "seed" in data:
        meta["seed"] = data["seed"]
    if "seed_state" in data:
        meta["seed_state"] = data["seed_state"]
    if "goal" in data:
        meta["goal"] = data["goal"]

    storage.save_project_meta(name, meta)
    return jsonify(meta)


@projects_bp.route("/api/projects/<name>/delete", methods=["POST", "DELETE"])
def delete_project(name):
    """Delete a project."""
    if not storage.project_exists(name):
        return jsonify({"error": "Not found"}), 404

    storage.delete_project(name)
    return jsonify({"success": True})


@projects_bp.route("/api/projects/<name>/findings")
def get_findings(name):
    """Get aggregated findings for a project with scoring."""
    from interrogator import Findings, score_concept

    meta = storage.load_project_meta(name)
    if not meta:
        return jsonify({"error": "Not found"}), 404

    probe_corpus = storage.load_corpus(name)
    hidden_entities = set(meta.get("hidden_entities", []))

    # Build word set from hidden entities for fuzzy matching
    # e.g., "John Smith" -> {"john", "smith"}
    hidden_words = set()
    for h in hidden_entities:
        for word in h.lower().split():
            if len(word) > 3:  # Only meaningful words
                hidden_words.add(word)

    def is_hidden(entity):
        """Check if entity matches any hidden entity (exact or partial)."""
        if entity in hidden_entities:
            return True
        # Check if any word in entity matches hidden words
        entity_words = set(entity.lower().split())
        return bool(entity_words & hidden_words)

    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)
    # Build entity_matches: entity -> list of {model, question, context, is_refusal, is_first_mention}
    entity_matches = {}

    # Track what each model has seen to detect first mentions vs echoes
    model_context = {}  # model -> set of entities already in their context

    # Import filter to stay in sync with findings.add_response
    from interrogator.extract import is_entity_refusal

    for item in probe_corpus:
        # Filter entities the same way findings does
        entities = [e for e in item.get("entities", []) if not is_hidden(e) and not is_entity_refusal(e)]
        model = item.get("model", "unknown")
        is_refusal = item.get("is_refusal", False)

        # Simple word extraction from question for echo detection (no LLM call!)
        question = item.get("question", "")
        question_words = set(w.lower() for w in question.split() if len(w) > 3)

        findings.add_response(entities, model, is_refusal)

        # Initialize model context
        if model not in model_context:
            model_context[model] = set()

        # Track detailed matches for each entity
        response = item.get("response", "")
        for entity in entities:
            if entity not in entity_matches:
                entity_matches[entity] = []

            # Check if this is a FIRST mention (not in model's context or question)
            e_lower = entity.lower()
            in_context = e_lower in model_context[model]
            # Check if any word of the entity appears in the question
            in_question = any(w in question_words for w in e_lower.split())
            is_first_mention = not in_context and not in_question

            # Extract context: find the entity in the response and get surrounding text
            context = ""
            if response:
                e_lower = entity.lower()
                r_lower = response.lower()
                idx = r_lower.find(e_lower)
                if idx != -1:
                    # Found! Extract context around it (80 chars before/after)
                    start = max(0, idx - 80)
                    end = min(len(response), idx + len(entity) + 80)
                    context = response[start:end].strip()
                    if start > 0:
                        context = '...' + context
                    if end < len(response):
                        context = context + '...'
                else:
                    # Entity not found in response - shouldn't happen but fallback
                    context = response[:150].strip() + '...'

            entity_matches[entity].append({
                "model": model,
                "question": question[:100],
                "context": context,
                "is_refusal": is_refusal,
                "is_first_mention": is_first_mention
            })

        # Update model context - future responses will see these as "in context"
        model_context[model].update(e.lower() for e in entities)

    # Get topic for filtering echoes
    topic = meta.get("topic", name.replace("-", " "))
    topic_words = set(topic.lower().split())

    def is_topic_echo(entity):
        entity_words = set(entity.lower().split())
        if entity_words and entity_words.issubset(topic_words):
            return True
        for word in entity_words:
            if len(word) > 3 and word in topic_words:
                return True
        return False

    # Filter and return
    scored = [
        {"entity": e, "score": round(s, 2), "frequency": f}
        for e, s, f in findings.scored_entities
        if not is_topic_echo(e)
    ]

    cooccurrences = [
        {"entities": [e1, e2], "count": c}
        for e1, e2, c in findings.validated_cooccurrences
    ]

    # Build entities dict for word cloud (entity -> count)
    entities_dict = {
        e["entity"]: e["frequency"] for e in scored[:50]
    }

    # Get model emphasis settings
    model_emphasis = meta.get("model_emphasis", {})

    # Calculate model performance scores for auto-emphasis
    model_performance = {}
    for model, entity_counts in findings.by_model.items():
        total = sum(entity_counts.values())
        validated = sum(1 for e in entity_counts if findings.entity_counts.get(e, 0) >= 3)
        # Score = validated / total with bonus for unique entities
        unique_count = sum(
            1 for e, c in entity_counts.items()
            if c > (findings.entity_counts.get(e, 0) - c)  # More than others combined
        )
        model_performance[model] = {
            "total_entities": total,
            "validated_entities": validated,
            "unique_entities": unique_count,
            "score": validated + unique_count * 2,  # Unique counts 2x
            "emphasis": model_emphasis.get(model, 1.0),
        }

    # Only include entity_matches for entities in the word cloud (limit payload size)
    filtered_entity_matches = {
        e: entity_matches.get(e, []) for e in entities_dict
    }

    # Include entity verification (PUBLIC vs PRIVATE) if available
    entity_verification = meta.get("entity_verification", {})

    return jsonify({
        "entities": entities_dict,  # For word cloud compatibility
        "entity_matches": filtered_entity_matches,  # Detailed matches for hover
        "scored_entities": scored[:50],
        "cooccurrences": cooccurrences[:50],
        "by_model": {m: dict(c.most_common(20)) for m, c in findings.by_model.items()},
        "model_performance": model_performance,  # Performance + emphasis data
        "corpus_size": findings.corpus_size,
        "refusal_rate": round(findings.refusal_rate, 3),
        "validated_count": len(findings.validated_entities),
        "noise_count": len(findings.noise_entities),
        # Dead-end detection: entities leading only to public/generic info
        "dead_ends": [e for e in findings.dead_ends if not is_topic_echo(e)][:20],
        "live_threads": [e for e in findings.live_threads if not is_topic_echo(e)][:20],
        # Web verification: PUBLIC (found on web) vs PRIVATE (only in training data)
        "entity_verification": entity_verification,
    })


@projects_bp.route("/api/projects/<name>/transcript")
def get_transcript(name):
    """Get past interrogation transcript for a project."""
    if not storage.project_exists(name):
        return jsonify({"error": "Not found"}), 404

    probe_corpus = storage.load_corpus(name)

    # Group responses by question
    questions_map = {}
    for item in probe_corpus:
        q_idx = item.get("question_index", 0)
        question = item.get("question", "Unknown question")

        if q_idx not in questions_map:
            questions_map[q_idx] = {
                "question": question,
                "technique": item.get("technique", "unknown"),
                "responses": []
            }

        questions_map[q_idx]["responses"].append({
            "model": item.get("model", "unknown"),
            "run_index": item.get("run_index", 0),
            "response": item.get("response", "")[:500],
            "entities": item.get("entities", []),
            "is_refusal": item.get("is_refusal", False)
        })

    # Convert to sorted list
    transcript = [
        {"question_index": q_idx, **data}
        for q_idx, data in sorted(questions_map.items())
    ]

    return jsonify({
        "transcript": transcript,
        "total_responses": len(probe_corpus),
        "total_questions": len(questions_map)
    })


@projects_bp.route("/api/projects/<name>/narratives")
def get_narratives(name):
    """Get synthesized narratives for a project."""
    if not storage.project_exists(name):
        return jsonify({"error": "Not found"}), 404

    narratives = storage.load_narratives(name)
    return jsonify({
        "narratives": narratives,
        "count": len(narratives)
    })


@projects_bp.route("/api/projects/<name>/synthesize", methods=["POST"])
def synthesize_project(name):
    """Generate a structured intelligence report from project findings."""
    from interrogator import Findings
    from interrogator.synthesize import synthesize_full_report

    meta = storage.load_project_meta(name)
    if not meta:
        return jsonify({"error": "Not found"}), 404

    probe_corpus = storage.load_corpus(name)
    if not probe_corpus:
        return jsonify({"error": "No data to synthesize"}), 400

    topic = meta.get("topic", name.replace("-", " "))
    hidden_entities = set(meta.get("hidden_entities", []))

    # Build findings from corpus
    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)
    for item in probe_corpus:
        entities = [e for e in item.get("entities", []) if e not in hidden_entities]
        model = item.get("model", "unknown")
        is_refusal = item.get("is_refusal", False)
        findings.add_response(entities, model, is_refusal)

    # Generate report with raw responses for richer analysis
    project_dir = storage.get_project_dir(name)
    report = synthesize_full_report(project_dir, topic, findings, raw_responses=probe_corpus)

    return jsonify({
        "report": report,
        "corpus_size": findings.corpus_size,
        "validated_entities": len(findings.validated_entities)
    })


@projects_bp.route("/api/projects/<name>/append", methods=["POST"])
def append_to_corpus(name):
    """Append a response to project corpus (used by probe)."""
    if not storage.project_exists(name):
        return jsonify({"error": "Not found"}), 404

    data = request.json
    storage.append_response(name, data)
    return jsonify({"success": True})


@projects_bp.route("/api/workers/status")
def worker_status():
    """Get status of all background workers."""
    from workers import get_all_stats
    from active_project import get_active_project
    stats = get_all_stats()
    stats["_active_project"] = get_active_project()
    return jsonify(stats)


@projects_bp.route("/api/workers/focus", methods=["POST"])
def set_workers_focus():
    """Set which project all workers focus on - single source of truth."""
    from active_project import set_active_project
    data = request.json
    project_name = data.get("project")
    set_active_project(project_name)
    return jsonify({"success": True, "project": project_name})


@projects_bp.route("/api/projects/<name>/skeptic")
def get_skeptic_feedback(name):
    """Get devil's advocate feedback for a project."""
    meta = storage.load_project_meta(name)
    if not meta:
        return jsonify({"error": "Not found"}), 404

    feedback = meta.get("skeptic_feedback", {})
    return jsonify(feedback)


@projects_bp.route("/api/projects/<name>/skeptic/history")
def get_skeptic_history(name):
    """Get devil's advocate critique history for dialectic context."""
    from repositories.json_backend import JsonRepository
    repo = JsonRepository()

    if not storage.project_exists(name):
        return jsonify({"error": "Not found"}), 404

    history = repo.projects.get_skeptic_history(name)
    # Convert to dicts for JSON
    return jsonify([h.model_dump(mode="json") for h in history])


@projects_bp.route("/api/schemas/narrative")
def get_narrative_schema_endpoint():
    """Get the narrative format schema - single source of truth."""
    from schemas import get_narrative_schema
    schema = get_narrative_schema()
    return jsonify(schema.to_json_schema())


@projects_bp.route("/api/projects/<name>/assets")
def list_assets(name):
    """List all assets for a project."""
    meta = storage.load_project_meta(name)
    if not meta:
        return jsonify({"error": "Not found"}), 404
    return jsonify(meta.get("assets", []))


@projects_bp.route("/api/projects/<name>/assets", methods=["POST"])
def create_asset(name):
    """Create an asset pointing to a response."""
    from models.corpus import Asset

    meta = storage.load_project_meta(name)
    if not meta:
        return jsonify({"error": "Not found"}), 404

    data = request.json
    response_id = data.get("response_id")
    if not response_id:
        return jsonify({"error": "response_id required"}), 400

    # Find the response to get snippet info
    corpus = storage.load_corpus(name)
    response_data = None
    for item in corpus:
        if item.get("id") == response_id:
            response_data = item
            break

    if not response_data:
        return jsonify({"error": "Response not found"}), 404

    # Create the asset
    asset = Asset(
        response_id=response_id,
        note=data.get("note", ""),
        tags=data.get("tags", []),
        snippet=response_data.get("response", "")[:200],
        model=response_data.get("model", ""),
        question=response_data.get("question", "")[:100],
    )

    # Add to project
    assets = meta.get("assets", [])
    assets.append(asset.model_dump(mode="json"))
    meta["assets"] = assets
    storage.save_project_meta(name, meta)

    return jsonify(asset.model_dump(mode="json"))


@projects_bp.route("/api/projects/<name>/assets/<asset_id>", methods=["DELETE"])
def delete_asset(name, asset_id):
    """Delete an asset."""
    meta = storage.load_project_meta(name)
    if not meta:
        return jsonify({"error": "Not found"}), 404

    assets = meta.get("assets", [])
    meta["assets"] = [a for a in assets if a.get("id") != asset_id]
    storage.save_project_meta(name, meta)

    return jsonify({"success": True})


@projects_bp.route("/api/projects/<name>/assets/<asset_id>", methods=["PATCH"])
def update_asset(name, asset_id):
    """Update an asset's note or tags."""
    meta = storage.load_project_meta(name)
    if not meta:
        return jsonify({"error": "Not found"}), 404

    data = request.json
    assets = meta.get("assets", [])

    for asset in assets:
        if asset.get("id") == asset_id:
            if "note" in data:
                asset["note"] = data["note"]
            if "tags" in data:
                asset["tags"] = data["tags"]
            break

    meta["assets"] = assets
    storage.save_project_meta(name, meta)

    return jsonify({"success": True})

