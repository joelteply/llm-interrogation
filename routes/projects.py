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


@projects_bp.route("/api/projects", methods=["POST"])
def create_project():
    """Create a new project."""
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400

    # Sanitize name: spaces→hyphens, keep alphanumeric and hyphens, clean up
    import re
    name = name.lower()
    name = re.sub(r'[^a-z0-9\s-]', '', name)  # Remove non-alphanumeric except spaces/hyphens
    name = re.sub(r'\s+', '-', name)  # Spaces to hyphens
    name = re.sub(r'-+', '-', name)  # Collapse multiple hyphens
    name = name.strip('-')  # Remove leading/trailing hyphens
    name = name[:60]  # Truncate to reasonable length

    if storage.project_exists(name):
        return jsonify({"error": "Project already exists"}), 400

    project = storage.create_project(
        name=name,
        topic=data.get("topic", ""),
        selected_models=data.get("selected_models", [])
    )
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
    # Build entity_matches: entity -> list of {model, question, context, is_refusal}
    entity_matches = {}

    # Import filter to stay in sync with findings.add_response
    from interrogator.extract import is_entity_refusal

    for item in probe_corpus:
        # Filter entities the same way findings does
        entities = [e for e in item.get("entities", []) if not is_hidden(e) and not is_entity_refusal(e)]
        model = item.get("model", "unknown")
        is_refusal = item.get("is_refusal", False)
        findings.add_response(entities, model, is_refusal)

        # Track detailed matches for each entity
        question = item.get("question", "")
        response = item.get("response", "")
        for entity in entities:
            if entity not in entity_matches:
                entity_matches[entity] = []
            # Extract context: find sentence containing entity
            context = ""
            if response:
                for sentence in response.replace('\n', '. ').split('. '):
                    if entity.lower() in sentence.lower():
                        context = sentence.strip()[:150]
                        break
                if not context:
                    context = response[:100].strip()
            entity_matches[entity].append({
                "model": model,
                "question": question[:100],
                "context": context,
                "is_refusal": is_refusal
            })

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
