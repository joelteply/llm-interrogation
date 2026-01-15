"""
Synthesis and curation API routes.
"""

import json
import re
from datetime import datetime
from flask import jsonify, request

from config import PROJECTS_DIR, get_client
from interrogator import Findings, build_synthesis_prompt
from . import probe_bp
from .helpers import get_project_models, extract_json


@probe_bp.route("/api/synthesize", methods=["POST"])
def synthesize():
    """Synthesize findings into narrative."""
    data = request.json
    project_name = data.get("project")

    if not project_name:
        return jsonify({"error": "Project required"}), 400

    project_file = PROJECTS_DIR / f"{project_name}.json"
    if not project_file.exists():
        return jsonify({"error": "Project not found"}), 404

    with open(project_file) as f:
        project = json.load(f)

    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)
    hidden_entities = set(project.get("hidden_entities", []))

    for item in project.get("probe_corpus", []):
        entities = [e for e in item.get("entities", []) if e not in hidden_entities]
        findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

    if len(findings.validated_entities) < 3:
        return jsonify({"error": "Need at least 3 validated entities"}), 400

    topic = project.get("topic", project_name.replace("-", " "))
    synthesis_prompt = build_synthesis_prompt(topic, findings)

    # Single source of truth
    selected_models = get_project_models(project_name)
    if not selected_models:
        selected_models = ["groq/llama-3.3-70b-versatile"]
    client, cfg = get_client(selected_models[0])

    try:
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        narrative = resp.choices[0].message.content

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


@probe_bp.route("/api/generate-theory", methods=["POST"])
def generate_theory():
    """Explicitly generate working theory for a project - for testing/manual trigger."""
    data = request.json
    project_name = data.get("project")

    if not project_name:
        return jsonify({"error": "Project required"}), 400

    project_file = PROJECTS_DIR / f"{project_name}.json"
    if not project_file.exists():
        return jsonify({"error": "Project not found"}), 404

    with open(project_file) as f:
        project = json.load(f)

    # Build findings from corpus
    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)
    topic = project.get("topic", project_name.replace("-", " "))

    for item in project.get("probe_corpus", []):
        entities = item.get("entities", [])
        findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

    if len(findings.scored_entities) < 3:
        return jsonify({"error": f"Need at least 3 scored entities, have {len(findings.scored_entities)}"}), 400

    # Single source of truth
    selected_models = get_project_models(project_name)
    if not selected_models:
        selected_models = ["groq/llama-3.3-70b-versatile"]
    client, cfg = get_client(selected_models[0])

    # Build entity string
    ent_str = ", ".join([f"{e} ({f}x)" for e, _, f in findings.scored_entities[:15]])
    print(f"[THEORY] Generating for topic='{topic}' with entities: {ent_str[:200]}...")

    prompt = f"""Analyze LEAKED TRAINING DATA from language models about: {topic}

ENTITIES EXTRACTED (frequency = signal strength): {ent_str}

Write like a newspaper reporter. OUTPUT FORMAT:

HEADLINE: [Punchy news headline. Name the key person/org/project. Be specific, be provocative.]
SUBHEAD: [1-2 sentences with specifics - who, what, when, where. Example: "Internal documents reveal CERDEC developed encryption management system in 2018 partnership."]

CLAIMS:
• [Specific fact with names/dates]
• [Another specific fact]

NEXT: [What to investigate]

CRITICAL: HEADLINE should grab attention and name names."""

    try:
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1500
        )
        narrative = resp.choices[0].message.content.strip()
        print(f"[THEORY] Generated ({len(narrative)} chars): {narrative[:200]}...")

        # Save to project with timestamp
        project["narrative"] = narrative
        project["working_theory"] = narrative
        project["narrative_updated"] = datetime.now().isoformat()
        project["updated"] = datetime.now().isoformat()
        with open(project_file, 'w') as f:
            json.dump(project, f, indent=2)

        print(f"[THEORY] *** SUCCESS *** Saved narrative to {project_name}")
        return jsonify({"narrative": narrative, "entity_count": len(findings.scored_entities)})

    except Exception as e:
        import traceback
        print(f"[THEORY ERROR] {e}")
        print(f"[THEORY ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@probe_bp.route("/api/auto-curate", methods=["POST"])
def auto_curate():
    """LLM curator - restructures data: merge duplicates, create relations, ban garbage, develop hunches."""
    data = request.json
    project_name = data.get("project")

    if not project_name:
        return jsonify({"error": "Project required"}), 400

    project_file = PROJECTS_DIR / f"{project_name}.json"
    if not project_file.exists():
        return jsonify({"error": "Project not found"}), 404

    with open(project_file) as f:
        project = json.load(f)

    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)
    topic = project.get("topic", project_name.replace("-", " "))

    for item in project.get("probe_corpus", []):
        entities = item.get("entities", [])
        findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

    # Get existing curations and theory
    existing_relations = project.get("relations", [])
    existing_hunches = project.get("hunches", [])
    working_theory = project.get("working_theory", "")

    # Format existing knowledge with confidence
    relations_str = "; ".join([
        f"{r['subject']} {r['predicate']} {r['object']} (conf:{r.get('confidence', 50)}%)"
        for r in existing_relations[:10]
    ]) if existing_relations else "none yet"

    # Build FULL context for semantic reasoning
    probe_corpus = project.get("probe_corpus", [])

    # All actual Q&A pairs (the raw evidence)
    qa_pairs = []
    for item in probe_corpus[:30]:  # Sample of actual evidence
        if not item.get("is_refusal"):
            q = item.get("question", "")[:100]
            r = item.get("response", "")[:200].replace("\n", " ")
            ents = item.get("entities", [])
            if q and r:
                qa_pairs.append(f"Q: {q}\nA: {r}\nEntities: {ents}")
    evidence_str = "\n---\n".join(qa_pairs[:15]) if qa_pairs else "none"

    # Co-occurrences (what appears together = semantic relationships)
    cooccur_str = "\n".join([f"  {e1} + {e2} (seen together {c}x)" for e1, e2, c in findings.validated_cooccurrences[:20]]) if findings.validated_cooccurrences else "none"

    # Dead ends and live threads
    dead_ends = findings.dead_ends[:15]
    live_threads = findings.live_threads[:15]

    # Build integrated evidence stream - each data point with full context
    evidence_stream = []
    for item in probe_corpus[:25]:
        if item.get("is_refusal"):
            continue
        q = item.get("question", "")[:80]
        r = item.get("response", "")[:150].replace("\n", " ")
        ents = item.get("entities", [])
        model = item.get("model", "unknown").split("/")[-1]
        if q and ents:
            evidence_stream.append(f"[{model}] Q: {q} → Extracted: {ents}")

    # Build integrated context
    integrated = f"""Investigation: {topic}

Working theory: {working_theory or "None yet"}
Known relations: {relations_str}
Live threads: {', '.join(live_threads) or 'none'}
Dead ends: {', '.join(dead_ends) or 'none'}

Evidence stream:
{chr(10).join(evidence_stream[:20])}

Entity co-occurrences:
{cooccur_str}

Top entities: {', '.join([f'{e}({f})' for e, _, f in findings.scored_entities[:30]])}"""

    prompt = f"""{integrated}

Analyze the evidence above for "{topic}".

CURATE the entity list:
- BAN garbage: generic words (Weekly, Final, Requirements), job titles (Project Manager, Software Developer), corporate speak (Stakeholder, Deliverables), partial phrases (If Jason, Verb To)
- PROMOTE signal: real names, companies, project codenames, specific dates

Return JSON with ACTUAL entity names from the list above:
{{"merge":[{{"canonical":"actual_name","aliases":["variant1"]}}],"relations":[{{"subject":"Person","predicate":"worked_at","object":"Company","confidence":80}}],"ban":["Weekly","Final","Project Manager","Requirements"],"promote":["Nerdy Deeds","Threat Intelligence"],"working_theory":"2-3 sentences connecting the evidence"}}"""

    # Pick a model for curation - prefer llama/deepseek (less restrictive), avoid GPT-4/Claude (often refuse)
    selected_models = get_project_models(project_name)
    curation_model = None

    # Priority: groq llama > deepseek > any non-GPT4/Claude > first available
    for m in (selected_models or []):
        if 'llama' in m.lower() or 'deepseek' in m.lower():
            curation_model = m
            break

    if not curation_model:
        for m in (selected_models or []):
            if 'gpt-4' not in m.lower() and 'claude' not in m.lower():
                curation_model = m
                break

    if not curation_model and selected_models:
        curation_model = selected_models[0]  # Last resort: use whatever is available

    if not curation_model:
        return jsonify({"error": "No models available for curation"}), 400

    print(f"[CURATE] Using model: {curation_model}")
    client, cfg = get_client(curation_model)

    try:
        print(f"[CURATE] Calling AI with cfg: {cfg}")
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2000
        )

        text = resp.choices[0].message.content
        print(f"[CURATE] Got response ({len(text)} chars): {text[:200]}...")
        json_str = extract_json(text, 'object')
        if not json_str:
            print(f"[CURATE ERROR] No valid JSON in response")
            return jsonify({"error": f"AI did not return valid JSON: {text[:300]}"}), 500

        result = json.loads(json_str)

        # APPLY CURATIONS TO DATA
        changes = {"merged": 0, "banned": 0, "relations_added": 0}

        # 1. Apply merges to probe_corpus
        merges = result.get("merge", [])
        for merge in merges:
            canonical = merge.get("canonical", "")
            aliases = merge.get("aliases", [])
            if canonical and aliases:
                for item in project.get("probe_corpus", []):
                    new_entities = []
                    for e in item.get("entities", []):
                        if e in aliases:
                            if canonical not in new_entities:
                                new_entities.append(canonical)
                            changes["merged"] += 1
                        else:
                            new_entities.append(e)
                    item["entities"] = new_entities

        # 2. Remove banned entities from probe_corpus
        ban_set = set(result.get("ban", []))
        for item in project.get("probe_corpus", []):
            original_len = len(item.get("entities", []))
            item["entities"] = [e for e in item.get("entities", []) if e not in ban_set]
            changes["banned"] += original_len - len(item["entities"])

        # 3. Store/update/sever relations based on confidence
        new_relations = result.get("relations", [])
        for rel in new_relations:
            subj, pred, obj = rel.get("subject"), rel.get("predicate"), rel.get("object")
            confidence = rel.get("confidence", 50)

            if not all([subj, pred, obj]):
                continue

            # Find existing relation
            existing_idx = None
            for i, er in enumerate(existing_relations):
                if er.get("subject") == subj and er.get("predicate") == pred and er.get("object") == obj:
                    existing_idx = i
                    break

            if confidence == 0:
                # SEVER: remove relation
                if existing_idx is not None:
                    existing_relations.pop(existing_idx)
                    changes["relations_added"] -= 1  # Actually severed
            elif existing_idx is not None:
                # UPDATE: adjust confidence
                existing_relations[existing_idx]["confidence"] = confidence
            else:
                # CREATE: new relation
                existing_relations.append({"subject": subj, "predicate": pred, "object": obj, "confidence": confidence})
                changes["relations_added"] += 1

        project["relations"] = existing_relations

        # 4. Update working theory
        new_theory = result.get("working_theory", "")
        if new_theory:
            project["working_theory"] = new_theory

        # 5. Update promoted entities
        promoted = set(project.get("promoted_entities", []))
        promoted.update(result.get("promote", []))
        project["promoted_entities"] = list(promoted)

        # Save changes
        project["updated"] = datetime.now().isoformat()
        with open(project_file, 'w') as f:
            json.dump(project, f, indent=2)

        return jsonify({
            "merge": merges,
            "relations": new_relations,
            "ban": result.get("ban", []),
            "promote": result.get("promote", []),
            "working_theory": new_theory,
            "changes": changes
        })

    except Exception as e:
        import traceback
        print(f"[CURATE ERROR] Exception: {e}")
        print(f"[CURATE ERROR] Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


def auto_curate_inline(topic: str, findings: Findings, hidden: set, promoted: list, project_name: str = None) -> dict:
    """Auto-curate entities during cycle - ban noise, promote promising.

    Uses intelligent analysis to identify:
    - Garbage: generic words LLMs capitalize (Weekly, Updates, Meeting)
    - Noise: terms unrelated to the topic
    - Signal: actual names, projects, dates relevant to topic
    """
    # Single source of truth
    selected_models = get_project_models(project_name) if project_name else []
    if not selected_models:
        return {"ban": [], "promote": []}
    client, cfg = get_client(selected_models[0])

    # Show entities with counts
    entities_str = ', '.join([
        f'{e}({f})' for e, _, f in findings.scored_entities[:50]
        if e not in hidden
    ])

    prompt = f"""Aggressively curate entities for investigation: "{topic}"

ENTITY LIST: {entities_str}

BE AGGRESSIVE. Most of these are garbage. Only a few are real signal.

BAN (return in "ban") - be harsh:
- Generic words that LLMs capitalize: Weekly, Final, Initial, Meeting, Update, Report, Summary
- Job titles: Project Manager, Software Developer, Analyst, Designer, Engineer, Director
- Corporate speak: Stakeholder, Requirements, Deliverables, Milestones, Budget, Resources
- Partial/broken phrases: "If Jason", "The Project", "Some Kind", sentence fragments
- Meta words: Lists, Refined, Details, Information, Various, Several, Multiple
- Common words wrongly capitalized: However, Additionally, Furthermore, Specifically
- Vague time refs: Recently, Previously, Earlier, Later, Eventually
- Single letters or numbers without context
- Anything that's clearly not a SPECIFIC name/place/company

CONSOLIDATE (return in "merge") - group duplicates:
- Same person different formats: ["John Smith", "J. Smith", "Smith, John"] → canonical: "John Smith"
- Company variants: ["Acme Corp", "Acme", "Acme Corporation"] → canonical: "Acme Corp"
- Partial matches that are clearly the same entity

PROMOTE (return in "promote") - only STRONG signal:
- Full names of actual people (first + last)
- Specific company/org names
- Project codenames or product names
- Specific dates/years with context

When in doubt, BAN IT. Better to remove noise than keep garbage.

JSON only (use actual entity names from list):
{{"ban": ["Weekly", "Project Manager", "The Project", "However"], "merge": [{{"canonical": "John Smith", "aliases": ["J. Smith"]}}], "promote": ["Nerdy Deeds"]}}"""

    try:
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )

        text = resp.choices[0].message.content
        json_str = extract_json(text, 'object')
        if json_str:
            return json.loads(json_str)
        print(f"[WARN] auto_curate_inline: no JSON found in response: {text[:200]}")
    except Exception as e:
        print(f"[ERROR] auto_curate_inline failed: {e}")
    return {"ban": [], "promote": []}
