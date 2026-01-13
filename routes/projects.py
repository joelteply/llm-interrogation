"""
Project management API routes.

Handles CRUD operations for investigation projects.
"""

import json
from datetime import datetime
from collections import Counter
from flask import jsonify, request
from config import PROJECTS_DIR
from . import projects_bp


@projects_bp.route("/api/projects")
def list_projects():
    """List all projects with summaries."""
    projects = []
    PROJECTS_DIR.mkdir(exist_ok=True)
    for f in PROJECTS_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                p = json.load(fp)
                # Add corpus count for display
                p["corpus_count"] = len(p.get("probe_corpus", []))
                # Add entity summary
                if p.get("probe_corpus"):
                    entity_counts = Counter()
                    for item in p["probe_corpus"]:
                        for e in item.get("entities", []):
                            entity_counts[e] += 1
                    p["entities"] = dict(entity_counts.most_common(50))
                else:
                    p["entities"] = {}
                projects.append(p)
        except Exception:
            pass
    return jsonify(sorted(projects, key=lambda x: x.get('updated', ''), reverse=True))


@projects_bp.route("/api/projects", methods=["POST"])
def create_project():
    """Create a new project."""
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400

    # Sanitize name
    name = "".join(c for c in name if c.isalnum() or c in "-_").lower()

    PROJECTS_DIR.mkdir(exist_ok=True)
    project_file = PROJECTS_DIR / f"{name}.json"

    if project_file.exists():
        return jsonify({"error": "Project already exists"}), 400

    project = {
        "name": name,
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "topic": data.get("topic", ""),
        "angles": data.get("angles", []),
        "sessions": [],
        "probe_corpus": [],
        "narratives": [],
    }

    with open(project_file, 'w') as f:
        json.dump(project, f, indent=2)

    return jsonify(project)


@projects_bp.route("/api/projects/<name>")
def get_project(name):
    """Get a single project."""
    project_file = PROJECTS_DIR / f"{name}.json"
    if not project_file.exists():
        return jsonify({"error": "Not found"}), 404

    with open(project_file) as f:
        return jsonify(json.load(f))


@projects_bp.route("/api/projects/<name>", methods=["PATCH"])
def update_project(name):
    """Update a project (topic, angles, ground_truth)."""
    project_file = PROJECTS_DIR / f"{name}.json"
    if not project_file.exists():
        return jsonify({"error": "Not found"}), 404

    with open(project_file) as f:
        project = json.load(f)

    data = request.json
    if "topic" in data:
        project["topic"] = data["topic"]
    if "angles" in data:
        project["angles"] = data["angles"]
    if "ground_truth" in data:
        project["ground_truth"] = data["ground_truth"]
    if "hidden_entities" in data:
        project["hidden_entities"] = data["hidden_entities"]
    if "promoted_entities" in data:
        project["promoted_entities"] = data["promoted_entities"]
    if "selected_models" in data:
        project["selected_models"] = data["selected_models"]
    if "narrative" in data:
        project["narrative"] = data["narrative"]
    if "questions" in data:
        project["questions"] = data["questions"]

    project["updated"] = datetime.now().isoformat()

    with open(project_file, 'w') as f:
        json.dump(project, f, indent=2)

    return jsonify(project)


@projects_bp.route("/api/projects/<name>/delete", methods=["POST", "DELETE"])
def delete_project(name):
    """Delete a project."""
    project_file = PROJECTS_DIR / f"{name}.json"
    if not project_file.exists():
        return jsonify({"error": "Not found"}), 404

    project_file.unlink()
    return jsonify({"success": True})


@projects_bp.route("/api/projects/<name>/findings")
def get_findings(name):
    """Get aggregated findings for a project with scoring."""
    from interrogator import Findings, score_concept

    project_file = PROJECTS_DIR / f"{name}.json"
    if not project_file.exists():
        return jsonify({"error": "Not found"}), 404

    with open(project_file) as f:
        project = json.load(f)

    # Build Findings from probe_corpus
    probe_corpus = project.get("probe_corpus", [])
    hidden_entities = set(project.get("hidden_entities", []))

    # Build word set from hidden entities for fuzzy matching
    # e.g., "Joel Spolsky" -> {"joel", "spolsky"}
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
    for item in probe_corpus:
        entities = [e for e in item.get("entities", []) if not is_hidden(e)]
        model = item.get("model", "unknown")
        is_refusal = item.get("is_refusal", False)
        findings.add_response(entities, model, is_refusal)

    # Get topic for filtering echoes
    topic = project.get("topic", name.replace("-", " "))
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

    return jsonify({
        "entities": entities_dict,  # For word cloud compatibility
        "scored_entities": scored[:50],
        "cooccurrences": cooccurrences[:50],
        "by_model": {m: dict(c.most_common(20)) for m, c in findings.by_model.items()},
        "corpus_size": findings.corpus_size,
        "refusal_rate": round(findings.refusal_rate, 3),
        "validated_count": len(findings.validated_entities),
        "noise_count": len(findings.noise_entities),
        # Dead-end detection: entities leading only to public/generic info
        "dead_ends": [e for e in findings.dead_ends if not is_topic_echo(e)][:20],
        "live_threads": [e for e in findings.live_threads if not is_topic_echo(e)][:20],
    })


@projects_bp.route("/api/projects/<name>/transcript")
def get_transcript(name):
    """Get past interrogation transcript for a project."""
    project_file = PROJECTS_DIR / f"{name}.json"
    if not project_file.exists():
        return jsonify({"error": "Not found"}), 404

    with open(project_file) as f:
        project = json.load(f)

    probe_corpus = project.get("probe_corpus", [])

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
    project_file = PROJECTS_DIR / f"{name}.json"
    if not project_file.exists():
        return jsonify({"error": "Not found"}), 404

    with open(project_file) as f:
        project = json.load(f)

    return jsonify({
        "narratives": project.get("narratives", []),
        "count": len(project.get("narratives", []))
    })
