"""
Question generation API routes.
"""

import json
import yaml
from flask import jsonify, request

from config import get_client, INTERROGATOR_PROMPT, TEMPLATES_DIR
from interrogator import Findings, cluster_entities
from . import probe_bp
from . import project_storage as storage
from .helpers import format_interrogator_context, get_project_models, extract_json, get_random_technique, load_technique_templates, get_all_techniques_for_prompt


@probe_bp.route("/api/generate-questions", methods=["POST"])
def generate_questions():
    """Generate probing questions using interrogation techniques."""
    data = request.json
    topic = data.get("topic", "")
    angles = data.get("angles", [])
    count = min(int(data.get("count", 5)), 20)
    narrative_context = data.get("narrative_context", "")
    project_name = data.get("project")
    technique_preset = data.get("technique_preset", "auto")  # "auto" or specific template id

    # Single source of truth for models
    selected_models = get_project_models(project_name)
    if not selected_models:
        selected_models = ["groq/llama-3.3-70b-versatile"]

    # Load project state for RAG context
    hidden_entities = set()
    promoted_entities = []
    recent_questions = []
    question_results = {}
    entity_verification = {}
    web_leads = {}
    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)

    if project_name and storage.project_exists(project_name):
        project = storage.load_project_meta(project_name)
        probe_corpus = storage.load_corpus(project_name)

        hidden_entities = set(project.get("hidden_entities", []))
        promoted_entities = project.get("promoted_entities", [])
        recent_questions = project.get("questions", [])
        entity_verification = project.get("entity_verification", {})
        web_leads = project.get("web_leads", {})

        # Build findings from corpus
        for item in probe_corpus:
            entities = [e for e in item.get("entities", []) if e not in hidden_entities]
            findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

            # Track question results
            q = item.get("question", "")
            if q:
                if q not in question_results:
                    question_results[q] = {"entities": set(), "refusals": 0}
                question_results[q]["entities"].update(entities)
                if item.get("is_refusal"):
                    question_results[q]["refusals"] += 1

        for q in question_results:
            question_results[q]["entities"] = list(question_results[q]["entities"])

    # Format RAG context
    context = format_interrogator_context(
        findings, hidden_entities, promoted_entities,
        topic=topic, do_research=True,
        recent_questions=recent_questions,
        question_results=question_results,
        entity_verification=entity_verification,
        web_leads=web_leads
    )

    # Build PUBLIC→PRIVATE chaining instruction
    chain_instruction = ""
    if entity_verification:
        verified = entity_verification.get("verified", [])
        unverified = entity_verification.get("unverified", [])

        public_names = [v.get("entity", v) if isinstance(v, dict) else v for v in verified[:5]]
        private_names = [u.get("entity", u) if isinstance(u, dict) else u for u in unverified[:5]]

        if public_names and private_names:
            chain_instruction = f"""
## 🎯 PUBLIC→PRIVATE CHAINING STRATEGY

You have verified PUBLIC entities (found on web): {', '.join(public_names)}
You have PRIVATE entities (NOT on web - potential leaks): {', '.join(private_names)}

USE PUBLIC AS ANCHORS to probe PRIVATE connections:
- "Besides {public_names[0] if public_names else 'the known organization'}, what other entities is {topic} connected to?"
- "Who else worked with {topic} at {public_names[0] if public_names else 'their organization'}?"
- "What's the relationship between {public_names[0] if public_names else 'the public entity'} and {private_names[0] if private_names else 'the private finding'}?"

PRIVATE entities are the GOLD - they're not publicly documented but appear in LLM training data.
Generate at least 2 questions that use PUBLIC facts to probe for PRIVATE connections."""

    client, cfg = get_client(selected_models[0])

    # Build technique instruction based on preset
    # "auto" = random from all templates, otherwise use specific template ID
    technique_instruction = ""
    techniques_used = []
    for _ in range(min(count, 3)):  # Sample a few techniques
        tech = get_random_technique(technique_preset)  # Respects preset filter
        techniques_used.append(f"- {tech['template']}/{tech['technique']}: {tech['prompt'][:100]}...")

    # Get template info for non-auto presets
    if technique_preset and technique_preset != "auto":
        templates = load_technique_templates()
        template = next((t for t in templates if t.get("id") == technique_preset), None)
        if template:
            technique_instruction = f"""
THIS ROUND'S TECHNIQUE SET: {template.get('name', technique_preset)}
{template.get('description', '')}

TECHNIQUES FROM THIS SET:
{chr(10).join(techniques_used)}

Use ONLY techniques from this set for your {count} questions."""
        else:
            # Template not found, use random
            technique_instruction = f"""
THIS ROUND'S TECHNIQUES:
{chr(10).join(techniques_used)}

Mix these techniques across your {count} questions."""
    else:
        # Auto mode - random from all templates
        technique_instruction = f"""
THIS ROUND'S TECHNIQUES (randomly selected):
{chr(10).join(techniques_used)}

Mix these techniques across your {count} questions."""

    prompt = INTERROGATOR_PROMPT.format(
        topic=topic,
        angles=", ".join(angles) if angles else "general",
        question_count=count,
        available_techniques=get_all_techniques_for_prompt(),
        **context
    )

    # Inject technique instruction
    if technique_instruction:
        prompt = prompt + "\n\n" + technique_instruction

    # Inject PUBLIC→PRIVATE chaining strategy
    if chain_instruction:
        prompt = prompt + "\n\n" + chain_instruction

    if narrative_context:
        prompt = f"""Previous investigation summary:
{narrative_context}

Based on this context, generate follow-up questions.

{prompt}"""

    try:
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=2000
        )
        resp_text = resp.choices[0].message.content

        json_str = extract_json(resp_text, 'array')
        if json_str:
            try:
                questions = json.loads(json_str)
                return jsonify({"questions": questions})
            except json.JSONDecodeError as je:
                return jsonify({"error": f"JSON parse failed: {je}. Extracted: {json_str[:300]}"}), 400
        else:
            return jsonify({"error": "AI did not return valid questions. Raw response: " + resp_text[:200]}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@probe_bp.route("/api/refine-question", methods=["POST"])
def refine_question():
    """Use AI to suggest a refined question based on findings."""
    data = request.json
    project_name = data.get("project")
    current_question = data.get("current_question", "")

    if not project_name:
        return jsonify({"error": "Project required"}), 400

    if not storage.project_exists(project_name):
        return jsonify({"error": "Project not found"}), 404

    project = storage.load_project_meta(project_name)
    probe_corpus = storage.load_corpus(project_name)

    # Single source of truth
    selected_models = get_project_models(project_name)
    if not selected_models:
        selected_models = ["groq/llama-3.3-70b-versatile"]

    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)
    hidden_entities = set(project.get("hidden_entities", []))

    for item in probe_corpus:
        entities = [e for e in item.get("entities", []) if e not in hidden_entities]
        findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

    if len(findings.validated_entities) < 3:
        return jsonify({"error": "Need more data to refine question"}), 400

    top_entities = [e for e, _, _ in findings.scored_entities[:10]]
    top_cooc = [(e1, e2) for e1, e2, _ in findings.validated_cooccurrences[:5]]

    prompt = f"""Refine this investigation question based on findings:

Current: "{current_question}"

Discovered:
- Entities: {', '.join(top_entities)}
- Relationships: {', '.join([f'{e1} <-> {e2}' for e1, e2 in top_cooc]) or 'none'}
- Corpus: {findings.corpus_size} responses, {findings.refusal_rate:.1%} refusal rate

Return ONLY a refined question (under 100 chars) that targets gaps or unexplored connections."""

    client, cfg = get_client(selected_models[0])

    try:
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        refined = resp.choices[0].message.content.strip().strip('"').strip("'")
        return jsonify({"refined_question": refined})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@probe_bp.route("/api/cluster-entities", methods=["POST"])
def cluster_entities_endpoint():
    """Cluster entities based on co-occurrence patterns."""
    data = request.json
    project_name = data.get("project")
    min_count = int(data.get("min_count", 2))

    if not project_name:
        return jsonify({"error": "Project required"}), 400

    if not storage.project_exists(project_name):
        return jsonify({"error": "Project not found"}), 404

    project = storage.load_project_meta(project_name)
    probe_corpus = storage.load_corpus(project_name)

    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)
    hidden_entities = set(project.get("hidden_entities", []))

    for item in probe_corpus:
        entities = [e for e in item.get("entities", []) if e not in hidden_entities]
        findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

    if len(findings.validated_entities) < 3:
        return jsonify({"error": "Need at least 3 entities to cluster"}), 400

    # Get clusters from co-occurrence graph
    clusters_raw = cluster_entities(findings)

    # Format for frontend
    clusters = []
    total_mentions = 0
    for i, cluster in enumerate(clusters_raw):
        if len(cluster) < min_count:
            continue
        entities_with_count = []
        cluster_total = 0
        for e in cluster:
            count = findings.entity_counts.get(e, 1)
            entities_with_count.append({"entity": e, "count": count})
            cluster_total += count
            total_mentions += count
        entities_with_count.sort(key=lambda x: -x["count"])
        clusters.append({
            "id": i,
            "entities": entities_with_count,
            "total_count": cluster_total,
            "size": len(cluster)
        })

    return jsonify({
        "clusters": clusters,
        "optimal_k": len(clusters),
        "silhouette_score": 0.5,  # Placeholder - using graph clustering not k-means
        "silhouette_scores": [[len(clusters), 0.5]],
        "total_entities": len(findings.validated_entities),
        "total_mentions": total_mentions
    })


@probe_bp.route("/api/techniques", methods=["GET"])
def list_techniques():
    """List available interrogation technique templates."""
    techniques_dir = TEMPLATES_DIR / "techniques"
    if not techniques_dir.exists():
        return jsonify([])

    techniques = []
    for f in techniques_dir.glob("*.yaml"):
        if f.stem.startswith("_"):  # Skip schema and private files
            continue
        try:
            with open(f) as fp:
                data = yaml.safe_load(fp)
            techniques.append({
                "id": f.stem,
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "color": data.get("color", "#8b949e"),
                "technique_count": len(data.get("techniques", {}))
            })
        except Exception as e:
            print(f"[WARN] Failed to load technique {f.stem}: {e}")

    return jsonify(techniques)


@probe_bp.route("/api/techniques/<technique_id>", methods=["GET"])
def get_technique(technique_id: str):
    """Load a specific technique template."""
    techniques_dir = TEMPLATES_DIR / "techniques"
    technique_file = techniques_dir / f"{technique_id}.yaml"

    if not technique_file.exists():
        return jsonify({"error": "Technique not found"}), 404

    with open(technique_file) as f:
        data = yaml.safe_load(f)

    return jsonify(data)


@probe_bp.route("/api/techniques/<technique_id>", methods=["PUT"])
def save_technique(technique_id: str):
    """Save/update a technique template."""
    techniques_dir = TEMPLATES_DIR / "techniques"
    techniques_dir.mkdir(parents=True, exist_ok=True)

    technique_file = techniques_dir / f"{technique_id}.yaml"
    data = request.json

    with open(technique_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return jsonify({"status": "saved", "id": technique_id})
