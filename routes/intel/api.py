"""
Intelligence System API

Endpoints for entity management, categorization, drilling, and analysis.
"""

import json
from pathlib import Path
from flask import request, jsonify, Response

from . import intel_bp
from .entities import EntityStore, Entity, make_entity_id
from .categorize import categorize_entity, categorize_entities
from .drill import drill_entity, DrillProtocol
from .confidence import calculate_confidence, confidence_label, prioritize_for_drill
from .graph import EntityGraph, analyze_graph


def get_store(project_name: str) -> EntityStore:
    """Get entity store for a project."""
    project_dir = Path(f"projects/{project_name}")
    return EntityStore(project_dir)


@intel_bp.route("/api/intel/<project_name>/entities", methods=["GET"])
def list_entities(project_name: str):
    """List all entities for a project."""
    store = get_store(project_name)

    # Optional filters
    category = request.args.get("category")
    min_confidence = float(request.args.get("min_confidence", 0))
    echo_only = request.args.get("echo_chambers") == "true"

    entities = store.all()

    if category:
        entities = [e for e in entities if e.category == category]

    if min_confidence > 0:
        entities = [e for e in entities if e.confidence >= min_confidence]

    if echo_only:
        entities = store.echo_chambers()

    # Sort by confidence
    entities = sorted(entities, key=lambda e: -e.confidence)

    return jsonify({
        "count": len(entities),
        "entities": [e.to_dict() for e in entities]
    })


@intel_bp.route("/api/intel/<project_name>/entities/<entity_id>", methods=["GET"])
def get_entity(project_name: str, entity_id: str):
    """Get a single entity with full details."""
    store = get_store(project_name)
    entity = store.get(entity_id)

    if not entity:
        return jsonify({"error": "Entity not found"}), 404

    return jsonify(entity.to_dict())


@intel_bp.route("/api/intel/<project_name>/categorize", methods=["POST"])
def categorize(project_name: str):
    """
    Categorize entities via web search (parallel).

    Body: {
        "entity_ids": [...] (optional, defaults to all unprocessed)
        "limit": 50
    }

    Returns SSE stream of progress.
    """
    store = get_store(project_name)
    data = request.get_json() or {}

    entity_ids = data.get("entity_ids")
    limit = data.get("limit", 50)

    # Get project topic
    project_file = Path(f"projects/{project_name}/project.json")
    topic = ""
    if project_file.exists():
        with open(project_file) as f:
            proj = json.load(f)
            topic = proj.get("topic", "")

    if entity_ids:
        entities = [store.get(eid) for eid in entity_ids if store.get(eid)]
    else:
        entities = [e for e in store.all() if e.category == "UNPROCESSED"]

    def generate():
        from concurrent.futures import ThreadPoolExecutor, as_completed

        to_process = entities[:limit]
        all_texts = [e.text for e in entities]

        yield f"data: {json.dumps({'type': 'start', 'count': len(to_process)})}\n\n"

        def process_one(entity):
            context = [t for t in all_texts if t != entity.text][:5]
            return categorize_entity(entity, context, topic)

        stats = {"corroborated": 0, "uncorroborated": 0, "contradicted": 0}
        done = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_one, e): e for e in to_process}

            for future in as_completed(futures):
                entity = futures[future]
                try:
                    category, web_results = future.result()
                    entity.category = category
                    entity.web_results = web_results
                    entity.confidence = calculate_confidence(entity)
                    stats[category.lower()] += 1
                    done += 1

                    yield f"data: {json.dumps({'type': 'progress', 'entity': entity.text, 'category': category, 'done': done, 'total': len(to_process)})}\n\n"
                except Exception as e:
                    done += 1
                    yield f"data: {json.dumps({'type': 'error', 'entity': entity.text, 'error': str(e)})}\n\n"

        store.save()
        yield f"data: {json.dumps({'type': 'complete', 'stats': stats})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@intel_bp.route("/api/intel/<project_name>/drill", methods=["POST"])
def drill(project_name: str):
    """
    Run drill protocol on entities.

    Body: {
        "entity_ids": [...] (optional, defaults to prioritized uncorroborated)
        "limit": 10
    }

    Returns SSE stream of drill progress.
    """
    store = get_store(project_name)
    data = request.get_json() or {}

    entity_ids = data.get("entity_ids")
    limit = data.get("limit", 10)

    # Get project topic
    project_file = Path(f"projects/{project_name}/project.json")
    topic = ""
    if project_file.exists():
        with open(project_file) as f:
            proj = json.load(f)
            topic = proj.get("topic", "")

    def generate():
        nonlocal entity_ids

        if entity_ids:
            entities = [store.get(eid) for eid in entity_ids if store.get(eid)]
        else:
            # Auto-prioritize
            entities = prioritize_for_drill(store.all())

        yield f"data: {json.dumps({'type': 'start', 'count': min(len(entities), limit)})}\n\n"

        for i, entity in enumerate(entities[:limit]):
            yield f"data: {json.dumps({'type': 'drilling', 'entity': entity.text, 'index': i})}\n\n"

            # Get the model that originated this claim
            model = entity.originated_from.model if entity.originated_from else "groq/llama-3.3-70b-versatile"

            # Run drill
            scores = drill_entity(entity, model, topic)
            entity.drill_scores = scores

            # Recalculate confidence
            entity.confidence = calculate_confidence(entity)

            yield f"data: {json.dumps({'type': 'drilled', 'entity': entity.text, 'scores': scores.to_dict(), 'confidence': entity.confidence, 'label': confidence_label(entity.confidence)})}\n\n"

        store.save()
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@intel_bp.route("/api/intel/<project_name>/graph", methods=["GET"])
def graph(project_name: str):
    """Get entity graph for visualization."""
    store = get_store(project_name)
    analysis = analyze_graph(store)
    return jsonify(analysis)


@intel_bp.route("/api/intel/<project_name>/summary", methods=["GET"])
def summary(project_name: str):
    """Get summary of intelligence for a project."""
    store = get_store(project_name)
    entities = store.all()

    # Category counts
    categories = {"CORROBORATED": 0, "UNCORROBORATED": 0, "CONTRADICTED": 0, "UNPROCESSED": 0}
    for e in entities:
        categories[e.category] = categories.get(e.category, 0) + 1

    # Confidence distribution
    high = [e for e in entities if e.confidence >= 0.7]
    medium = [e for e in entities if 0.5 <= e.confidence < 0.7]
    low = [e for e in entities if 0.3 <= e.confidence < 0.5]
    very_low = [e for e in entities if e.confidence < 0.3]

    # Echo chambers
    echo_chambers = store.echo_chambers()

    # Top entities by confidence
    top = sorted(entities, key=lambda e: -e.confidence)[:20]

    return jsonify({
        "total_entities": len(entities),
        "categories": categories,
        "confidence_distribution": {
            "high": len(high),
            "medium": len(medium),
            "low": len(low),
            "very_low": len(very_low)
        },
        "echo_chambers": len(echo_chambers),
        "needs_drill": len([e for e in entities if e.category == "UNCORROBORATED" and not e.drill_scores]),
        "top_entities": [
            {
                "text": e.text,
                "category": e.category,
                "confidence": e.confidence,
                "label": confidence_label(e.confidence),
                "sources": e.independent_sources,
                "is_echo_chamber": e.is_echo_chamber
            }
            for e in top
        ]
    })


@intel_bp.route("/api/intel/<project_name>/import", methods=["POST"])
def import_from_corpus(project_name: str):
    """
    Import entities from existing corpus.jsonl into the intel system.

    This bridges the old probe system with the new intel system.
    """
    store = get_store(project_name)
    corpus_file = Path(f"projects/{project_name}/corpus.jsonl")

    if not corpus_file.exists():
        return jsonify({"error": "No corpus file found"}), 404

    imported = 0
    with open(corpus_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                entities = data.get("entities", [])
                response_text = data.get("response", "")
                question_text = data.get("question", "")
                model = data.get("model", "unknown")
                q_idx = data.get("question_index", 0)

                for ent_text in entities:
                    if not ent_text or len(ent_text) < 2:
                        continue

                    store.add_entity(
                        text=ent_text,
                        model=model,
                        response_id=f"corpus_{q_idx}_{hash(response_text) % 10000}",
                        question_id=f"q_{q_idx}",
                        question_text=question_text,
                        response_text=response_text
                    )
                    imported += 1

            except Exception as e:
                print(f"[import] Error: {e}")
                continue

    # Calculate initial confidence for all
    for entity in store.all():
        entity.confidence = calculate_confidence(entity)

    store.save()

    return jsonify({
        "status": "complete",
        "imported": imported,
        "total_entities": len(store.all())
    })


@intel_bp.route("/api/intel/<project_name>/report", methods=["GET"])
def report(project_name: str):
    """
    Generate intelligence report.

    Shows entities by confidence tier with full context.
    """
    store = get_store(project_name)
    entities = store.all()

    # Group by confidence tier
    tiers = {
        "HIGH": [],
        "MEDIUM": [],
        "LOW": [],
        "VERY LOW": []
    }

    for e in entities:
        label = confidence_label(e.confidence)
        tiers[label].append(e)

    # Sort each tier by confidence
    for tier in tiers.values():
        tier.sort(key=lambda e: -e.confidence)

    report = {
        "project": project_name,
        "total_entities": len(entities),
        "tiers": {}
    }

    for tier_name, tier_entities in tiers.items():
        report["tiers"][tier_name] = {
            "count": len(tier_entities),
            "entities": [
                {
                    "text": e.text,
                    "confidence": e.confidence,
                    "category": e.category,
                    "sources": e.independent_sources,
                    "mentions": e.total_mentions,
                    "is_echo_chamber": e.is_echo_chamber,
                    "drill_total": e.drill_scores.total if e.drill_scores else None,
                    "originated_from": e.originated_from.model if e.originated_from else None
                }
                for e in tier_entities[:50]  # Top 50 per tier
            ]
        }

    return jsonify(report)
