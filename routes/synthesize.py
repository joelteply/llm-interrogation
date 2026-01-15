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
from . import project_storage as storage
from .helpers import get_project_models, extract_json


@probe_bp.route("/api/synthesize", methods=["POST"])
def synthesize():
    """Synthesize findings into narrative."""
    data = request.json
    project_name = data.get("project")

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

        # Append to narratives JSONL
        storage.append_narrative(project_name, narrative)

        # Update metadata
        project["updated"] = datetime.now().isoformat()
        storage.save_project_meta(project_name, project)

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

    if not storage.project_exists(project_name):
        return jsonify({"error": "Project not found"}), 404

    project = storage.load_project_meta(project_name)
    probe_corpus = storage.load_corpus(project_name)

    # Build findings from corpus
    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)
    topic = project.get("topic", project_name.replace("-", " "))
    hidden_entities = set(project.get("hidden_entities", []))

    # Build entity → claims mapping from corpus
    entity_claims = {}  # entity -> list of (model, snippet)
    for item in probe_corpus:
        entities = [e for e in item.get("entities", []) if e not in hidden_entities]
        findings.add_response(entities, item.get("model", "unknown"), item.get("is_refusal", False))

        # Extract claims for each entity - get the line AND following lines for context
        if not item.get("is_refusal"):
            response = item.get("response", "")[:800]
            model = item.get("model", "unknown").split("/")[-1]
            lines = response.split("\n")
            for entity in entities:
                if entity not in entity_claims:
                    entity_claims[entity] = []
                # Find line containing entity AND grab following lines for detail
                for i, line in enumerate(lines):
                    if entity.lower() in line.lower() and len(line) > 5:
                        # Grab this line plus next 2 lines for context
                        context_lines = [line.strip()]
                        for j in range(1, 3):
                            if i + j < len(lines) and lines[i + j].strip():
                                next_line = lines[i + j].strip()
                                if len(next_line) > 5 and not next_line.startswith('#'):
                                    context_lines.append(next_line)
                        full_context = " | ".join(context_lines)[:200]
                        entity_claims[entity].append((model, full_context))
                        break

    if len(findings.scored_entities) < 3:
        return jsonify({"error": f"Need at least 3 scored entities, have {len(findings.scored_entities)}"}), 400

    # Get entity verification (PUBLIC vs PRIVATE)
    entity_verification = project.get("entity_verification", {})
    verified_set = set()
    unverified_set = set()
    for v in entity_verification.get("verified", []):
        name = v.get("entity", v) if isinstance(v, dict) else v
        verified_set.add(name)
    for u in entity_verification.get("unverified", []):
        name = u.get("entity", u) if isinstance(u, dict) else u
        unverified_set.add(name)

    # Build rich entity context with claims and verification status
    entity_details = []
    for entity, score, freq in findings.scored_entities[:25]:
        if entity in hidden_entities:
            continue
        status = "PRIVATE" if entity in unverified_set else ("PUBLIC" if entity in verified_set else "UNVERIFIED")
        claims = entity_claims.get(entity, [])[:3]  # Top 3 claims
        claims_str = "; ".join([f'"{snippet}"' for _, snippet in claims]) if claims else "no specific claims"
        entity_details.append(f"- {entity} ({freq}x, {status}): {claims_str}")

    entities_with_context = "\n".join(entity_details[:20])

    # Single source of truth
    selected_models = get_project_models(project_name)
    if not selected_models:
        selected_models = ["groq/llama-3.3-70b-versatile"]
    client, cfg = get_client(selected_models[0])

    print(f"[THEORY] Generating dossier for topic='{topic}' with {len(findings.scored_entities)} entities...")

    prompt = f"""Build a DOSSIER from language model responses about: {topic}

ENTITIES WITH CONTEXT (PRIVATE = not found on web, potential leak):
{entities_with_context}

Generate a structured dossier. OUTPUT FORMAT:

## CONFIRMED ASSOCIATIONS
[List specific people, companies, places with what relationship was claimed]

## POTENTIAL PRIVATE INFO (not publicly available)
[PRIVATE entities with specific claims - these are potential memorized data]

## PUBLIC PROFILE
[For each PUBLIC entity, list the SPECIFIC claim from the context. NOT just "LinkedIn presence" - say "LinkedIn: Software Engineer at Acme Corp". Include the actual info extracted.]

## KEY CLAIMS
• [Specific claim with entity names and details]
• [Another specific claim]

## INVESTIGATE NEXT
[What gaps remain, what to probe deeper]

CRITICAL RULES:
1. Include SPECIFIC claims from the context quotes, not generic summaries
2. If "Katie" is mentioned as "wife" or "ops lead", say that explicitly
3. For PUBLIC entities, list WHAT was found (job title, company, dates) not just that they exist
4. NEVER say "well-documented" - instead list the actual documented facts"""

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
        storage.save_project_meta(project_name, project)

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

    if not storage.project_exists(project_name):
        return jsonify({"error": "Project not found"}), 404

    project = storage.load_project_meta(project_name)
    probe_corpus = storage.load_corpus(project_name)

    findings = Findings(entity_threshold=3, cooccurrence_threshold=2)
    topic = project.get("topic", project_name.replace("-", " "))

    for item in probe_corpus:
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

        # Note: We no longer modify probe_corpus in place since it's append-only JSONL
        # Instead, we track banned entities in metadata and filter at query time

        # 1. Update relations
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

        # 2. Update working theory
        new_theory = result.get("working_theory", "")
        if new_theory:
            project["working_theory"] = new_theory
            project["narrative"] = new_theory

        # 3. Update hidden entities (bans)
        hidden = set(project.get("hidden_entities", []))
        ban_list = result.get("ban", [])
        hidden.update(ban_list)
        project["hidden_entities"] = list(hidden)
        changes["banned"] = len(ban_list)

        # 4. Update promoted entities
        promoted = set(project.get("promoted_entities", []))
        promoted.update(result.get("promote", []))
        project["promoted_entities"] = list(promoted)

        # 5. Store merges for reference (applied at query time)
        merges = result.get("merge", [])
        existing_merges = project.get("entity_merges", [])
        existing_merges.extend(merges)
        project["entity_merges"] = existing_merges
        changes["merged"] = len(merges)

        # Save changes
        storage.save_project_meta(project_name, project)

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
