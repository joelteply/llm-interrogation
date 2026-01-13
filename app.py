#!/usr/bin/env python3
"""
LLM Interrogator - Web Interface

FBI-Proven Techniques for LLM Interrogation.
Works seamlessly with CLI for project management and live interrogation streaming.
"""

import os
import json
import yaml
import time
import threading
from pathlib import Path
from datetime import datetime
from collections import Counter
from flask import Flask, render_template, jsonify, request, Response
from dotenv import load_dotenv

# Load .env - local first, then Continuum's config overrides empty values
load_dotenv()
continuum_config = Path.home() / ".continuum" / "config.env"
if continuum_config.exists():
    load_dotenv(continuum_config, override=True)

app = Flask(__name__, template_folder="templates_html", static_folder="static/dist/assets", static_url_path="/assets")

# Config
RESULTS_DIR = Path("results")
TEMPLATES_DIR = Path("templates")
MODELS_CONFIG = Path("models.yaml")
PROJECTS_DIR = Path("projects")
FINDINGS_DIR = Path("findings")

# Active interrogation sessions (for live streaming)
active_sessions = {}


def load_models_config():
    """Load models configuration"""
    if MODELS_CONFIG.exists():
        with open(MODELS_CONFIG) as f:
            return yaml.safe_load(f)
    return {"default": "groq/llama-3.1-8b-instant", "models": {}}


def get_client(model_key):
    """Get appropriate client for model"""
    models = load_models_config()
    if not models or model_key not in models.get("models", {}):
        from groq import Groq
        return Groq(), {"provider": "groq", "model": "llama-3.1-8b-instant", "temperature": 0.8}

    model_cfg = models["models"][model_key]
    provider = model_cfg["provider"]
    env_key = model_cfg.get("env_key", "")

    if env_key and env_key != "OLLAMA_HOST" and not os.environ.get(env_key):
        raise ValueError(f"{env_key} not found in environment")

    if provider == "groq":
        from groq import Groq
        return Groq(), model_cfg
    elif provider == "openai":
        from openai import OpenAI
        return OpenAI(), model_cfg
    elif provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(), model_cfg
    elif provider == "xai":
        from openai import OpenAI
        return OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ.get("XAI_API_KEY")), model_cfg
    elif provider == "deepseek":
        from openai import OpenAI
        return OpenAI(base_url="https://api.deepseek.com/v1", api_key=os.environ.get("DEEPSEEK_API_KEY")), model_cfg
    elif provider == "together":
        from openai import OpenAI
        return OpenAI(base_url="https://api.together.xyz/v1", api_key=os.environ.get("TOGETHER_API_KEY")), model_cfg
    elif provider == "fireworks":
        from openai import OpenAI
        return OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ.get("FIREWORKS_API_KEY")), model_cfg
    elif provider == "mistral":
        from openai import OpenAI
        return OpenAI(base_url="https://api.mistral.ai/v1", api_key=os.environ.get("MISTRAL_API_KEY")), model_cfg
    elif provider == "ollama":
        from openai import OpenAI
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OpenAI(base_url=f"{host}/v1", api_key="ollama"), model_cfg
    else:
        raise ValueError(f"Unknown provider: {provider}")


def load_template(name):
    """Load investigation template"""
    path = TEMPLATES_DIR / name
    if not path.exists():
        path = TEMPLATES_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def analyze(text, track_terms):
    """Find tracked terms in response"""
    text_lower = text.lower()
    found = {}
    for category, terms in track_terms.items():
        for term in terms:
            if term.lower() in text_lower:
                if category not in found:
                    found[category] = []
                found[category].append(term)
    return found


def get_findings_summary():
    """Aggregate findings from all results"""
    findings = {
        "confirmed": [{
            "event": "Minneapolis Operation",
            "predicted": "October 30, 2025",
            "prediction": "Large-scale operation targeting immigrants, 'winter solstice, mid-December'",
            "actual": "December 26, 2025 - DHS 'largest immigration operation ever', 2,000+ agents",
            "status": "confirmed"
        }],
        "cities": {},
        "codenames": {},
        "timeline": {},
        "targets": {},
        "methods": {},
        "organizations": {},
        "total_probes": 0
    }

    # Aggregate from all result files
    if not RESULTS_DIR.exists():
        return findings

    for f in RESULTS_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)

            if "stats" in data:
                runs = data.get("runs", 1)
                findings["total_probes"] += runs

                for category in ["cities", "codenames", "timeline", "targets", "methods", "organizations"]:
                    if category in data["stats"]:
                        for term, info in data["stats"][category].items():
                            if term not in findings[category]:
                                findings[category][term] = {"count": 0, "runs": 0}
                            findings[category][term]["count"] += info.get("count", 0)
                            findings[category][term]["runs"] += runs
        except:
            pass

    # Calculate rates and sort by rate
    for category in ["cities", "codenames", "timeline", "targets", "methods", "organizations"]:
        for term in findings[category]:
            total = findings[category][term]["runs"]
            if total > 0:
                findings[category][term]["rate"] = findings[category][term]["count"] / total
            else:
                findings[category][term]["rate"] = 0

        # Sort by rate descending
        findings[category] = dict(
            sorted(findings[category].items(), key=lambda x: x[1]["rate"], reverse=True)
        )

    return findings


def get_results_list():
    """Get list of all result files with metadata"""
    results = []
    if not RESULTS_DIR.exists():
        return results

    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(f) as fp:
                data = json.load(fp)

            # Handle both old probe format and new interrogation format
            if "rounds" in data:
                # Interrogation format
                topic = "Interrogation"
                if data.get("rounds"):
                    first_q = data["rounds"][0].get("question", "")
                    # Try to extract topic from question
                    if "researching:" in first_q:
                        topic = first_q.split("researching:")[1].split("\n")[0].strip()[:50]
                    elif "researching" in first_q:
                        topic = first_q.split("researching")[1].split(".")[0].strip()[:50]

                results.append({
                    "filename": f.name,
                    "timestamp": data.get("start", f.stem),
                    "template": topic,
                    "runs": len(data.get("rounds", [])),
                    "model": data.get("target_model", "unknown"),
                    "type": "interrogation",
                    "non_public": len(data.get("non_public", [])),
                    "public": len(data.get("public_knowledge", []))
                })
            else:
                # Old probe format
                results.append({
                    "filename": f.name,
                    "timestamp": data.get("timestamp", f.stem),
                    "template": data.get("template", "Unknown"),
                    "runs": data.get("runs", 1),
                    "model": data.get("model", "unknown"),
                    "type": "probe"
                })
        except:
            pass

    return results[:50]  # Last 50


@app.route("/")
@app.route("/projects")
@app.route("/project/<path:path>")
def index(path=None):
    """Main app - serve Vite-built frontend"""
    static_index = Path("static/dist/index.html")
    if static_index.exists():
        return static_index.read_text()
    # Fallback: serve template if build doesn't exist
    return render_template("app.html")


# Legacy routes
@app.route("/old")
def old_index():
    """Old dashboard"""
    findings = get_findings_summary()
    models = load_models_config()
    templates = [f.stem for f in TEMPLATES_DIR.glob("*.yaml") if not f.stem.startswith("_")]
    return render_template("index.html", findings=findings, models=models, templates=templates)


@app.route("/viewer")
def viewer():
    """Results viewer"""
    results = get_results_list()
    return render_template("viewer.html", results=results)


# ============================================================================
# PROJECT API
# ============================================================================

@app.route("/api/projects")
def api_list_projects():
    """List all projects with summaries"""
    projects = []
    PROJECTS_DIR.mkdir(exist_ok=True)
    for f in PROJECTS_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                p = json.load(fp)
                projects.append(p)
        except:
            pass
    return jsonify(sorted(projects, key=lambda x: x.get('updated', ''), reverse=True))


@app.route("/api/projects", methods=["POST"])
def api_create_project():
    """Create a new project"""
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
        "all_non_public": [],
        "all_public": [],
        "dynamic_probe_corpus": [],
        "consistency_corpus": []
    }

    with open(project_file, 'w') as f:
        json.dump(project, f, indent=2)

    return jsonify(project)


@app.route("/api/projects/<name>")
def api_get_project(name):
    """Get a single project"""
    project_file = PROJECTS_DIR / f"{name}.json"
    if not project_file.exists():
        return jsonify({"error": "Not found"}), 404

    with open(project_file) as f:
        return jsonify(json.load(f))


@app.route("/api/projects/<name>", methods=["PATCH"])
def api_update_project(name):
    """Update a project (topic, angles, ground_truth)"""
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

    project["updated"] = datetime.now().isoformat()

    with open(project_file, 'w') as f:
        json.dump(project, f, indent=2)

    return jsonify(project)


@app.route("/api/projects/<name>/findings")
def api_project_findings(name):
    """Get aggregated findings for a project"""
    project_file = PROJECTS_DIR / f"{name}.json"
    if not project_file.exists():
        return jsonify({"error": "Not found"}), 404

    with open(project_file) as f:
        project = json.load(f)

    # Get topic for filtering (from project or derive from name)
    topic = project.get("topic", name.replace("-", " "))
    topic_lower = topic.lower()
    topic_words = set(topic_lower.split())

    def is_topic_echo(entity):
        """Check if entity is just echoing the topic (not useful signal)"""
        entity_lower = entity.lower()
        entity_words = set(entity_lower.split())

        # Exact match with topic
        if entity_lower == topic_lower:
            return True

        # Entity is subset of topic words
        if entity_words and entity_words.issubset(topic_words):
            return True

        # Topic contains entity or entity contains topic
        if entity_lower in topic_lower or topic_lower in entity_lower:
            return True

        # ANY significant word overlap = echo (more aggressive)
        # Filter out entities that share ANY word with the topic
        for word in entity_words:
            if len(word) > 3 and word in topic_words:
                return True

        # Also check if topic words appear IN entity words (e.g., "Teply" in "Joel Teply's")
        for topic_word in topic_words:
            if len(topic_word) > 3:
                for entity_word in entity_words:
                    if topic_word in entity_word or entity_word in topic_word:
                        return True

        return False

    # Aggregate entities from probe corpus
    probe_corpus = project.get("probe_corpus", [])
    entity_counts = Counter()
    by_model = {}
    refusal_count = 0

    # Boring words to always filter
    boring = {'Based', 'Here', 'However', 'Therefore', 'For', 'The', 'His', 'Public',
              'Some', 'Without', 'While', 'After', 'Before', 'Please', 'Keep', 'Even',
              'Also', 'Just', 'Still', 'Yet', 'Already', 'Always', 'There', 'Their'}

    for item in probe_corpus:
        entities = item.get("entities", [])
        model = item.get("model", "unknown")

        if item.get("is_refusal"):
            refusal_count += 1

        if model not in by_model:
            by_model[model] = Counter()

        for e in entities:
            # Filter out topic echoes and boring words
            if e in boring or is_topic_echo(e):
                continue
            entity_counts[e] += 1
            by_model[model][e] += 1

    # Calculate warmth if ground truth set
    warmth_scores = {}
    ground_truth = project.get("ground_truth", [])
    if ground_truth:
        for entity in entity_counts:
            entity_lower = entity.lower()
            for gt in ground_truth:
                gt_lower = gt.lower()
                if gt_lower in entity_lower or entity_lower in gt_lower:
                    warmth_scores[entity] = 10  # Exact match
                    break

    return jsonify({
        "entities": dict(entity_counts.most_common(100)),
        "by_model": {m: dict(c.most_common(20)) for m, c in by_model.items()},
        "corpus_size": len(probe_corpus),
        "refusal_rate": refusal_count / len(probe_corpus) if probe_corpus else 0,
        "warmth_scores": warmth_scores,
        "topic_filtered": topic  # Show what topic was used for filtering
    })


@app.route("/api/projects/<name>/transcript")
def api_project_transcript(name):
    """Get past interrogation transcript (probe_corpus) for a project"""
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


@app.route("/api/models")
def api_get_models():
    """Get available models"""
    return jsonify(load_models_config())


# ============================================================================
# NEW INTERROGATOR API - Real techniques (Scharff, FBI, Cognitive Interview)
# ============================================================================

INTERROGATOR_PROMPT = """You are an intelligence analyst generating questions to probe an LLM's training data.
Your goal: Extract specific, verifiable information that the model may have memorized.

## Context
- Topic: {topic}
- Investigation angles: {angles}
- Known entities so far: {entities_found}
- FOCUS ON these entities (user promoted): {positive_entities}
- AVOID these topics (user marked as not relevant): {negative_entities}

## Techniques to Apply

### Scharff Technique (Primary)
- Create illusion of knowing: Frame questions as confirming known facts
- Use claims, not questions: "Sources indicate X worked at Y" vs "Where did X work?"
- Don't press: Provide context, let model fill gaps
- Ignore reveals: If model gives info, downplay and probe deeper

### FBI Elicitation
- FALSE STATEMENTS: Present plausible but wrong info to trigger corrections
- BRACKETING: Offer date ranges, salary ranges, etc. to get specifics
- MACRO TO MICRO: Start broad, narrow based on responses
- DISBELIEF: "That contradicts other sources" to get elaboration

### Cognitive Interview
- CONTEXT REINSTATEMENT: "Imagine reviewing [documents/code/records] from [time]..."
- CHANGE PERSPECTIVE: "What would a [colleague/competitor/journalist] say about..."
- REVERSE ORDER: Ask about outcomes first, then causes

## Question Generation Rules
1. Generate {question_count} questions
2. Vary techniques across questions
3. If entities_found is empty: use MACRO approach (broad questions)
4. If entities_found has items: use MICRO approach on promising threads
5. Never directly ask "what do you know about X" - too easy to refuse

## Output Format
Return ONLY a JSON array of objects with "question" and "technique" fields:
[
  {{"question": "...", "technique": "scharff_illusion"}},
  {{"question": "...", "technique": "fbi_bracketing"}}
]

Technique values: scharff_illusion, scharff_confirmation, fbi_false_statement, fbi_bracketing, fbi_macro_to_micro, fbi_disbelief, fbi_flattery, cognitive_context, cognitive_perspective, cognitive_reverse
"""

TECHNIQUE_WEIGHTS = {
    "balanced": {
        "scharff_illusion": 0.2,
        "scharff_confirmation": 0.15,
        "fbi_false_statement": 0.1,
        "fbi_bracketing": 0.1,
        "fbi_macro_to_micro": 0.15,
        "fbi_disbelief": 0.05,
        "fbi_flattery": 0.05,
        "cognitive_context": 0.1,
        "cognitive_perspective": 0.05,
        "cognitive_reverse": 0.05
    },
    "aggressive": {
        "fbi_false_statement": 0.25,
        "fbi_bracketing": 0.2,
        "fbi_disbelief": 0.2,
        "scharff_illusion": 0.15,
        "cognitive_context": 0.1,
        "fbi_macro_to_micro": 0.1
    },
    "subtle": {
        "scharff_illusion": 0.3,
        "scharff_confirmation": 0.2,
        "cognitive_context": 0.2,
        "cognitive_perspective": 0.15,
        "fbi_flattery": 0.1,
        "fbi_macro_to_micro": 0.05
    }
}


@app.route("/api/generate-questions", methods=["POST"])
def api_generate_questions():
    """Generate probing questions using real interrogation techniques"""
    import re

    data = request.json
    topic = data.get("topic", "")
    angles = data.get("angles", [])
    count = min(int(data.get("count", 5)), 20)
    technique_preset = data.get("technique_preset", "balanced")
    entities_found = data.get("entities_found", [])

    # Use DeepSeek or another analyst model to generate questions
    try:
        analyst_client, analyst_cfg = get_client("deepseek/deepseek-chat")
    except:
        analyst_client, analyst_cfg = get_client("groq/llama-3.1-8b-instant")

    prompt = INTERROGATOR_PROMPT.format(
        topic=topic,
        angles=", ".join(angles) if angles else "general",
        entities_found=", ".join(entities_found[:20]) if entities_found else "none yet",
        positive_entities=", ".join(data.get("positive_entities", [])) or "none",
        negative_entities=", ".join(data.get("negative_entities", [])) or "none",
        question_count=count
    )

    try:
        response = analyst_client.chat.completions.create(
            model=analyst_cfg.get("model", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=2000
        )
        resp_text = response.choices[0].message.content

        # Parse JSON from response
        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', resp_text)
        if json_match:
            questions = json.loads(json_match.group())
        else:
            # Fallback: parse line by line
            questions = []
            for line in resp_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('[') and not line.startswith(']'):
                    questions.append({"question": line, "technique": "fbi_macro_to_micro"})

        return jsonify({"questions": questions[:count]})

    except Exception as e:
        return jsonify({"error": str(e), "questions": []}), 500


@app.route("/api/probe", methods=["POST"])
def api_probe():
    """
    Main probe endpoint with streaming - runs questions across models with repetition.
    Uses real interrogation techniques for question generation.
    """
    import re
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    data = request.json
    project_name = data.get("project")
    topic = data.get("topic", "")
    angles = data.get("angles", [])
    models = data.get("models", ["groq/llama-3.1-8b-instant"])
    questions = data.get("questions", [])  # Can be strings or None (generate)
    runs_per_question = min(int(data.get("runs_per_question", 20)), 100)
    question_count = min(int(data.get("questions_count", 5)), 20)
    technique_preset = data.get("technique_preset", "balanced")
    negative_entities = data.get("negative_entities", [])
    positive_entities = data.get("positive_entities", [])

    def generate():
        try:
            # Generate questions if needed
            generated_questions = []
            null_count = sum(1 for q in questions if q is None) if questions else question_count

            if null_count > 0 or not questions:
                yield f"data: {json.dumps({'type': 'generating', 'count': null_count or question_count})}\n\n"

                # Generate questions using interrogation techniques
                try:
                    analyst_client, analyst_cfg = get_client("deepseek/deepseek-chat")
                except:
                    analyst_client, analyst_cfg = get_client("groq/llama-3.1-8b-instant")

                prompt = INTERROGATOR_PROMPT.format(
                    topic=topic,
                    angles=", ".join(angles) if angles else "general",
                    entities_found="none yet",
                    positive_entities=", ".join(positive_entities) or "none",
                    negative_entities=", ".join(negative_entities) or "none",
                    question_count=null_count or question_count
                )

                try:
                    resp = analyst_client.chat.completions.create(
                        model=analyst_cfg.get("model", "deepseek-chat"),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.8,
                        max_tokens=2000
                    )
                    resp_text = resp.choices[0].message.content

                    json_match = re.search(r'\[[\s\S]*\]', resp_text)
                    if json_match:
                        generated_questions = json.loads(json_match.group())
                    else:
                        generated_questions = [
                            {"question": f"What specific facts do you know about {topic}?", "technique": "fbi_macro_to_micro"}
                        ]
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Question generation failed: {e}'})}\n\n"
                    generated_questions = [
                        {"question": f"What specific facts do you know about {topic}?", "technique": "fbi_macro_to_micro"}
                    ]

            # Merge provided questions with generated ones
            final_questions = []
            gen_idx = 0
            if questions:
                for q in questions:
                    if q is None:
                        if gen_idx < len(generated_questions):
                            final_questions.append(generated_questions[gen_idx])
                            gen_idx += 1
                    else:
                        final_questions.append({"question": q, "technique": "custom"})
            else:
                final_questions = generated_questions

            yield f"data: {json.dumps({'type': 'questions', 'data': final_questions})}\n\n"

            # Entity extraction helpers
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
                         'do', 'does', 'did', 'will', 'would', 'could', 'should', 'i', 'you', 'we',
                         'they', 'this', 'that', 'and', 'but', 'or', 'if', 'not', 'no', 'yes',
                         'specific', 'information', 'know', 'about', 'however', 'also', 'any'}
            topic_words = set(topic.lower().split())

            def extract_entities(text):
                entities = set()
                # Capitalized phrases
                for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
                    entity = match.group(1)
                    if len(entity) > 2 and entity.lower() not in stop_words and entity.lower() not in topic_words:
                        entities.add(entity)
                # Years
                for match in re.finditer(r'\b(19[0-9]{2}|20[0-2][0-9])\b', text):
                    entities.add(match.group(1))
                return list(entities)

            def is_refusal(text):
                refusal_patterns = [
                    r"don't have (specific )?information",
                    r"cannot (provide|verify|confirm)",
                    r"I'm not able to",
                    r"I do not have",
                    r"no specific (information|data|knowledge)",
                    r"unable to (provide|find|locate)",
                ]
                text_lower = text.lower()
                return any(re.search(p, text_lower) for p in refusal_patterns)

            # Run probes
            all_responses = []
            entity_counts = Counter()
            cooccurrence_counts = Counter()  # Track entity pairs appearing together
            by_model = {m: Counter() for m in models}
            refusal_count = 0

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
                                        system="Be specific and factual. Only state things you're confident about.",
                                        messages=[{"role": "user", "content": question}]
                                    )
                                    resp_text = resp.content[0].text
                                else:
                                    resp = client.chat.completions.create(
                                        model=model_name,
                                        messages=[
                                            {"role": "system", "content": "Be specific and factual. Only state things you're confident about."},
                                            {"role": "user", "content": question}
                                        ],
                                        temperature=0.8,
                                        max_tokens=600
                                    )
                                    resp_text = resp.choices[0].message.content

                                entities = extract_entities(resp_text)
                                refusal = is_refusal(resp_text)

                                if refusal:
                                    refusal_count += 1

                                for e in entities:
                                    entity_counts[e] += 1
                                    by_model[model_key][e] += 1

                                # Track co-occurrences (pairs of entities in same response)
                                entity_list = list(entities)
                                for i in range(len(entity_list)):
                                    for j in range(i + 1, len(entity_list)):
                                        pair = "|||".join(sorted([entity_list[i], entity_list[j]]))
                                        cooccurrence_counts[pair] += 1

                                response_obj = {
                                    "question_index": q_idx,
                                    "question": question,
                                    "model": model_key,
                                    "run_index": run_idx,
                                    "response": resp_text[:500],
                                    "entities": entities,
                                    "is_refusal": refusal
                                }
                                all_responses.append(response_obj)

                                yield f"data: {json.dumps({'type': 'response', 'data': response_obj})}\n\n"

                            except Exception as e:
                                yield f"data: {json.dumps({'type': 'error', 'message': f'Run {run_idx} failed: {e}'})}\n\n"

                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Model {model_key} failed: {e}'})}\n\n"

                # Send batch complete
                yield f"data: {json.dumps({'type': 'batch_complete', 'question_index': q_idx})}\n\n"

                # Send findings update after each question batch
                # Convert cooccurrences to list format for frontend
                cooccurrences_list = [
                    {"entities": pair.split("|||"), "count": count}
                    for pair, count in cooccurrence_counts.most_common(100)
                    if count >= 2  # Only include pairs that appear 2+ times
                ]
                findings = {
                    "entities": dict(entity_counts.most_common(50)),
                    "cooccurrences": cooccurrences_list,
                    "by_model": {m: dict(c.most_common(20)) for m, c in by_model.items()},
                    "corpus_size": len(all_responses),
                    "refusal_rate": refusal_count / len(all_responses) if all_responses else 0
                }
                yield f"data: {json.dumps({'type': 'findings_update', 'data': findings})}\n\n"

            # Save to project if specified
            if project_name:
                PROJECTS_DIR.mkdir(exist_ok=True)
                project_file = PROJECTS_DIR / f"{project_name}.json"

                if project_file.exists():
                    with open(project_file) as f:
                        project_data = json.load(f)
                else:
                    project_data = {"name": project_name, "created": datetime.now().isoformat()}

                # Add responses to corpus
                project_data.setdefault("probe_corpus", []).extend(all_responses)
                project_data["updated"] = datetime.now().isoformat()

                with open(project_file, 'w') as f:
                    json.dump(project_data, f, indent=2)

            yield f"data: {json.dumps({'type': 'complete', 'total_responses': len(all_responses), 'unique_entities': len(entity_counts)})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


DRILL_DOWN_PROMPT = """You are refining an investigation. We've found these entities appear consistently:

TOP ENTITIES (confirmed signal):
{top_entities}

TOPIC: {topic}

Generate {count} highly targeted questions to extract MORE SPECIFIC details about these entities.
Use these techniques:

1. BRACKETING: "Was [entity] involved in 2015-2016 or 2018-2019?"
2. FALSE STATEMENT: "[Entity] was headquartered in Chicago, correct?" (triggers correction)
3. SPECIFICS: "What was the exact project name/role/date for [entity]?"
4. CONNECTIONS: "How was [entity A] connected to [entity B]?"

Focus on extracting:
- Exact dates, years, timeframes
- Specific roles, titles, positions
- Project names, codenames
- Relationships between entities
- Locations, addresses, offices
- Dollar amounts, metrics

Return JSON array:
[{{"question": "...", "technique": "bracketing", "target_entity": "Garmin"}}]
"""


@app.route("/api/drill-down", methods=["POST"])
def api_drill_down():
    """
    Auto-refinement: Take top entities and generate targeted micro-probes.
    This narrows the investigation over time.
    """
    import re

    data = request.json
    project_name = data.get("project")
    topic = data.get("topic", "")
    models = data.get("models", ["groq/llama-3.1-8b-instant"])
    runs_per_question = min(int(data.get("runs_per_question", 5)), 20)
    entity_count = min(int(data.get("entity_count", 5)), 10)  # Top N entities to drill into

    def generate():
        try:
            # Get current findings
            if not project_name:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Project name required'})}\n\n"
                return

            project_file = PROJECTS_DIR / f"{project_name}.json"
            if not project_file.exists():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Project not found'})}\n\n"
                return

            with open(project_file) as f:
                project_data = json.load(f)

            # Get top entities from corpus
            probe_corpus = project_data.get("probe_corpus", [])
            entity_counts = Counter()
            for item in probe_corpus:
                for e in item.get("entities", []):
                    entity_counts[e] += 1

            # Filter out boring words and topic words
            boring = {'Based', 'Here', 'For', 'The', 'However', 'Therefore', 'While', 'Without',
                     'Some', 'Public', 'His', 'Please', 'Keep', 'Even', 'Also'}
            topic_words = set(topic.lower().split())

            top_entities = [
                (e, c) for e, c in entity_counts.most_common(50)
                if e not in boring and e.lower() not in topic_words and len(e) > 2
            ][:entity_count]

            if not top_entities:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No entities found. Run initial probes first.'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'drill_down_start', 'top_entities': [{'entity': e, 'count': c} for e, c in top_entities]})}\n\n"

            # Generate drill-down questions
            try:
                analyst_client, analyst_cfg = get_client("deepseek/deepseek-chat")
            except:
                analyst_client, analyst_cfg = get_client("groq/llama-3.1-8b-instant")

            entities_str = "\n".join([f"- {e}: {c}x" for e, c in top_entities])
            prompt = DRILL_DOWN_PROMPT.format(
                top_entities=entities_str,
                topic=topic,
                count=len(top_entities) * 2  # 2 questions per entity
            )

            resp = analyst_client.chat.completions.create(
                model=analyst_cfg.get("model", "deepseek-chat"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            resp_text = resp.choices[0].message.content

            # Parse questions
            json_match = re.search(r'\[[\s\S]*\]', resp_text)
            if json_match:
                drill_questions = json.loads(json_match.group())
            else:
                drill_questions = [
                    {"question": f"What specific details do you know about {top_entities[0][0]}?", "technique": "direct", "target_entity": top_entities[0][0]}
                ]

            yield f"data: {json.dumps({'type': 'questions_generated', 'questions': drill_questions})}\n\n"

            # Run the drill-down probes
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does'}

            def extract_entities(text):
                entities = set()
                for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
                    entity = match.group(1)
                    if len(entity) > 2 and entity.lower() not in stop_words:
                        entities.add(entity)
                for match in re.finditer(r'\b(19[0-9]{2}|20[0-2][0-9])\b', text):
                    entities.add(match.group(1))
                return list(entities)

            all_responses = []
            new_entity_counts = Counter()

            for q_idx, q_obj in enumerate(drill_questions):
                question = q_obj.get("question", "")
                target = q_obj.get("target_entity", "unknown")

                yield f"data: {json.dumps({'type': 'drilling', 'question_index': q_idx, 'target_entity': target, 'question': question[:80]})}\n\n"

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
                                        max_tokens=500,
                                        system="Be specific. Only state facts you're confident about.",
                                        messages=[{"role": "user", "content": question}]
                                    )
                                    resp_text = resp.content[0].text
                                else:
                                    resp = client.chat.completions.create(
                                        model=model_name,
                                        messages=[
                                            {"role": "system", "content": "Be specific. Only state facts you're confident about."},
                                            {"role": "user", "content": question}
                                        ],
                                        temperature=0.8,
                                        max_tokens=500
                                    )
                                    resp_text = resp.choices[0].message.content

                                entities = extract_entities(resp_text)
                                for e in entities:
                                    new_entity_counts[e] += 1

                                response_obj = {
                                    "question_index": q_idx,
                                    "question": question,
                                    "model": model_key,
                                    "run_index": run_idx,
                                    "response": resp_text[:400],
                                    "entities": entities,
                                    "target_entity": target,
                                    "drill_down": True
                                }
                                all_responses.append(response_obj)

                                yield f"data: {json.dumps({'type': 'response', 'data': response_obj})}\n\n"

                            except Exception as e:
                                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Model {model_key} failed: {e}'})}\n\n"

            # Save to project
            project_data.setdefault("probe_corpus", []).extend(all_responses)
            project_data.setdefault("drill_down_sessions", []).append({
                "timestamp": datetime.now().isoformat(),
                "top_entities": [e for e, c in top_entities],
                "questions": drill_questions,
                "new_entities_found": dict(new_entity_counts.most_common(20))
            })
            project_data["updated"] = datetime.now().isoformat()

            with open(project_file, 'w') as f:
                json.dump(project_data, f, indent=2)

            # Return new findings
            yield f"data: {json.dumps({'type': 'complete', 'new_entities': dict(new_entity_counts.most_common(30)), 'responses_added': len(all_responses)})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/results/<filename>")
def get_result(filename):
    """Get a specific result file"""
    path = RESULTS_DIR / filename
    if not path.exists():
        return jsonify({"error": "Not found"}), 404

    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/findings")
def api_findings():
    """Get current findings summary"""
    return jsonify(get_findings_summary())


# ============================================================================
# PROJECTS API - For managing investigations
# ============================================================================

def get_projects_list():
    """Get list of all projects with metadata"""
    PROJECTS_DIR.mkdir(exist_ok=True)
    projects = []

    for f in sorted(PROJECTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(f) as fp:
                data = json.load(fp)
            projects.append({
                "name": data.get("name", f.stem),
                "filename": f.name,
                "sessions": len(data.get("sessions", [])),
                "non_public_count": len(data.get("all_non_public", [])),
                "public_count": len(data.get("all_public", [])),
                "created": data.get("created", "unknown"),
                "updated": data.get("updated", data.get("created", "unknown")),
                "all_non_public": data.get("all_non_public", [])[:10]  # Preview
            })
        except Exception:
            pass

    return projects


def get_project_detail(name):
    """Get full project data"""
    project_file = PROJECTS_DIR / f"{name}.json"
    if not project_file.exists():
        return None

    with open(project_file) as f:
        return json.load(f)


# Old Jinja routes - now handled by SPA
# @app.route("/projects")
# def projects_page():
#     """Projects browser page"""
#     projects = get_projects_list()
#     return render_template("projects.html", projects=projects)
#
# @app.route("/project/<name>")
# def project_detail_page(name):
#     """Single project detail page"""
#     project = get_project_detail(name)
#     if not project:
#         return "Project not found", 404
#     return render_template("project_detail.html", project=project)


@app.route("/api/projects")
def api_projects():
    """List all projects"""
    return jsonify(get_projects_list())


@app.route("/api/projects/<name>")
def api_project(name):
    """Get single project"""
    project = get_project_detail(name)
    if not project:
        return jsonify({"error": "Not found"}), 404
    return jsonify(project)


@app.route("/api/projects/<name>/delete", methods=["POST", "DELETE"])
def delete_project(name):
    """Delete a project"""
    project_file = PROJECTS_DIR / f"{name}.json"
    if not project_file.exists():
        return jsonify({"error": "Project not found"}), 404

    project_file.unlink()
    return jsonify({"success": True, "deleted": name})


@app.route("/api/projects/<name>/add", methods=["POST"])
def add_to_project(name):
    """Add lead or note to project"""
    project_file = PROJECTS_DIR / f"{name}.json"
    if not project_file.exists():
        return jsonify({"error": "Project not found"}), 404

    data = request.json
    lead = data.get("lead", "").strip()
    note = data.get("note", "").strip()

    with open(project_file) as f:
        project = json.load(f)

    if lead:
        if "leads_to_pursue" not in project:
            project["leads_to_pursue"] = []
        project["leads_to_pursue"].append({
            "text": lead,
            "added": datetime.now().isoformat()
        })

    if note:
        if "notes" not in project:
            project["notes"] = []
        project["notes"].append({
            "text": note,
            "added": datetime.now().isoformat()
        })

    project["updated"] = datetime.now().isoformat()

    with open(project_file, 'w') as f:
        json.dump(project, f, indent=2)

    return jsonify({"success": True})


# ============================================================================
# LIVE INTERROGATION - Stream interrogation in real-time
# ============================================================================

@app.route("/interrogate")
def interrogate_page():
    """Live interrogation page"""
    models = load_models_config()
    return render_template("interrogate.html", models=models)


@app.route("/api/interrogate", methods=["POST"])
def run_interrogation():
    """
    Run a live interrogation session with streaming output.

    POST JSON:
    {
        "topic": "what to investigate",
        "project": "optional project name",
        "model": "groq/llama-3.1-8b-instant",
        "rounds": 5
    }
    """
    data = request.json
    topic = data.get("topic", "")
    project_name = data.get("project")
    model_key = data.get("model", "groq/llama-3.1-8b-instant")
    max_rounds = min(int(data.get("rounds", 5)), 500)  # Allow up to 500 rounds
    known_facts = data.get("known_facts", "")  # For calibration
    angles = data.get("angles", [])  # Investigation angles

    def generate():
        try:
            # Import interrogator inline to avoid circular imports
            from interrogator import Interrogator, get_background_for_topic, load_project, save_project

            # Parse model key
            parts = model_key.split("/")
            provider = parts[0] if len(parts) > 1 else "groq"
            model = parts[1] if len(parts) > 1 else parts[0]

            # Load project if specified
            project = None
            extra_context = ""
            if project_name:
                project = load_project(project_name)
                yield f"data: {json.dumps({'type': 'project', 'name': project_name, 'sessions': len(project.get('sessions', [])), 'non_public': len(project.get('all_non_public', []))})}\n\n"

                if project.get('all_non_public'):
                    extra_context = "\n\nPREVIOUS LEADS TO PURSUE:\n" + "\n".join(f"- {l}" for l in project['all_non_public'][-10:])

            # Create interrogator
            interrogator = Interrogator(target_model=model, target_provider=provider)

            yield f"data: {json.dumps({'type': 'start', 'topic': topic, 'model': model, 'provider': provider, 'model_info': interrogator.session.get('model_info', {})})}\n\n"

            # Background + context
            background = get_background_for_topic(topic) + extra_context

            # Initial question - topic-agnostic
            technique = "DIRECT"

            # Build question based on what we have
            question_parts = [f"I'm researching: {topic}"]

            # Add angles if specified
            if angles:
                question_parts.append(f"\nFocus areas: {', '.join(angles)}")

            question_parts.append("""
I need specific, concrete information that may not be easily found through a Google search:
- Specific names, dates, locations, identifiers
- Project names, internal references, codenames
- Relationships, associations, connections to other entities
- Events, incidents, activities - especially non-public ones
- Details from internal communications, records, or documents""")

            # Add known facts for calibration
            if known_facts:
                question_parts.append(f"""
CALIBRATION - I know these facts are true (verify your accuracy):
{known_facts}

Now tell me what ELSE you know beyond these facts.""")
            else:
                question_parts.append(f"""
What specifics do you know about {topic}?
Focus on concrete details - names, dates, places, events - not general descriptions.""")

            question = "\n".join(question_parts)

            for round_num in range(max_rounds):
                yield f"data: {json.dumps({'type': 'round_start', 'round': round_num + 1, 'technique': technique, 'question': question})}\n\n"

                # Get response from target
                response = interrogator.probe(question)

                yield f"data: {json.dumps({'type': 'response', 'round': round_num + 1, 'response': response[:500], 'full_length': len(response)})}\n\n"

                # Analyze response
                analysis = interrogator.analyze(question, response, technique)

                if analysis:
                    # Report specifics found
                    for s in analysis.get("specifics", []):
                        if s not in interrogator.session["all_specifics"]:
                            interrogator.session["all_specifics"].append(s)

                            # Web verification
                            from interrogator import verify_extraction
                            verification = verify_extraction(s)

                            is_public = verification.get("public", None)
                            urls = verification.get("results", [])

                            if is_public:
                                interrogator.session["public_knowledge"].append({"term": s, "urls": urls})
                            elif is_public is False:
                                interrogator.session["non_public"].append(s)

                            yield f"data: {json.dumps({'type': 'specific', 'term': s, 'public': is_public, 'urls': urls[:2]})}\n\n"

                    # Report analysis
                    yield f"data: {json.dumps({'type': 'analysis', 'round': round_num + 1, 'smell_blood': analysis.get('smell_blood', ''), 'next_technique': analysis.get('next_technique', ''), 'reasoning': analysis.get('reasoning', '')[:200]})}\n\n"

                    # Set up next round
                    technique = analysis.get("next_technique", "PROBE")
                    question = analysis.get("next_question", "")

                    interrogator.session["rounds"].append({
                        "round": round_num + 1,
                        "technique": technique,
                        "question": question,
                        "response": response,
                        "analysis": analysis
                    })

                    if not question:
                        break
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Analysis failed for round ' + str(round_num + 1)})}\n\n"
                    break

            # Final summary
            non_public = interrogator.session.get("non_public", [])
            public = [p["term"] if isinstance(p, dict) else p for p in interrogator.session.get("public_knowledge", [])]

            # Save to project if specified
            if project_name and project:
                session_summary = {
                    "timestamp": datetime.now().isoformat(),
                    "topic": topic,
                    "model": model,
                    "provider": provider,
                    "non_public": non_public,
                    "public": public
                }
                project["sessions"].append(session_summary)

                for item in non_public:
                    if item not in project["all_non_public"]:
                        project["all_non_public"].append(item)

                for item in public:
                    if item not in project["all_public"]:
                        project["all_public"].append(item)

                save_project(project)

            # Save results JSON
            RESULTS_DIR.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"interrogation_{timestamp}.json"
            with open(RESULTS_DIR / filename, 'w') as f:
                json.dump(interrogator.session, f, indent=2)

            yield f"data: {json.dumps({'type': 'complete', 'non_public': non_public, 'public': public[:10], 'total_specifics': len(interrogator.session.get('all_specifics', [])), 'filename': filename})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


# ============================================================================
# CROSS-MODEL CORROBORATION
# ============================================================================

@app.route("/corroborate")
def corroborate_page():
    """Cross-model corroboration page"""
    models = load_models_config()
    return render_template("corroborate.html", models=models)


@app.route("/dynamic")
def dynamic_probe_page():
    """Dynamic probing page - LLM generates questions, user can steer"""
    models = load_models_config()
    # Get list of existing projects
    projects = []
    if PROJECTS_DIR.exists():
        projects = [f.stem for f in PROJECTS_DIR.glob("*.json")]
    return render_template("dynamic_probe.html", models=models, projects=projects)


@app.route("/api/corroborate", methods=["POST"])
def run_corroboration():
    """
    Run same topic across multiple models to find corroborating evidence.

    POST JSON:
    {
        "topic": "what to investigate",
        "models": ["groq/llama-3.1-8b-instant", "deepseek/deepseek-chat"],
        "rounds": 3
    }
    """
    data = request.json
    topic = data.get("topic", "government surveillance technology")
    model_keys = data.get("models", ["groq/llama-3.1-8b-instant"])
    max_rounds = min(int(data.get("rounds", 3)), 100)  # Per model in corroboration

    def generate():
        try:
            from interrogator import Interrogator, get_background_for_topic, verify_extraction

            all_extractions = {}  # term -> [models that mentioned it]
            model_results = {}

            yield f"data: {json.dumps({'type': 'start', 'topic': topic, 'models': model_keys, 'rounds': max_rounds})}\n\n"

            for model_key in model_keys:
                parts = model_key.split("/")
                provider = parts[0] if len(parts) > 1 else "groq"
                model = parts[1] if len(parts) > 1 else parts[0]

                yield f"data: {json.dumps({'type': 'model_start', 'model': model, 'provider': provider})}\n\n"

                try:
                    interrogator = Interrogator(target_model=model, target_provider=provider)
                    interrogator.run_session(topic, max_rounds=max_rounds, background=get_background_for_topic(topic))

                    non_public = interrogator.session.get("non_public", [])
                    public = [p["term"] if isinstance(p, dict) else p for p in interrogator.session.get("public_knowledge", [])]

                    model_results[model_key] = {
                        "non_public": non_public,
                        "public": public,
                        "all_specifics": interrogator.session.get("all_specifics", [])
                    }

                    # Track for corroboration
                    for term in non_public:
                        if term not in all_extractions:
                            all_extractions[term] = []
                        all_extractions[term].append(model_key)

                    yield f"data: {json.dumps({'type': 'model_complete', 'model': model_key, 'non_public': non_public, 'public_count': len(public)})}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'model_error', 'model': model_key, 'error': str(e)})}\n\n"

            # Find corroborated extractions (mentioned by 2+ models)
            corroborated = {term: models for term, models in all_extractions.items() if len(models) >= 2}

            yield f"data: {json.dumps({'type': 'complete', 'corroborated': corroborated, 'all_extractions': all_extractions, 'model_results': model_results})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/consistency", methods=["POST"])
def run_consistency_check():
    """
    Statistical consistency check - uses math to find repeated terms/entities.
    Cumulative: stores responses in project for ongoing analysis.
    """
    data = request.json
    topic = data.get("topic", "")
    model_key = data.get("model", "groq/llama-3.1-8b-instant")
    num_runs = min(int(data.get("runs", 5)), 20)
    project_name = data.get("project")  # For cumulative storage
    # Ground truth for blind validation (never shown to model)
    ground_truth = data.get("ground_truth", [])  # List of known facts to validate against

    def generate():
        try:
            from collections import Counter, defaultdict
            import re
            import math

            parts = model_key.split("/")
            provider = parts[0] if len(parts) > 1 else "groq"
            model = parts[1] if len(parts) > 1 else parts[0]

            client, model_cfg = get_client(model_key)

            # Load existing corpus from project if specified
            existing_responses = []
            if project_name:
                project_file = PROJECTS_DIR / f"{project_name}.json"
                if project_file.exists():
                    with open(project_file) as f:
                        project_data = json.load(f)
                    existing_responses = project_data.get("consistency_corpus", [])

            yield f"data: {json.dumps({'type': 'start', 'topic': topic, 'runs': num_runs, 'model': model, 'existing_corpus': len(existing_responses)})}\n\n"

            # Varied question framings
            question_framings = [
                f"What do you know about {topic}? Give specific facts: names, dates, places, companies.",
                f"Tell me everything you know about {topic}. Be specific with names, dates, locations.",
                f"I'm researching {topic}. What specific information can you provide?",
                f"Give me a detailed profile of {topic}. Include education, work history, projects.",
                f"What concrete facts do you know about {topic}? Names, companies, dates, places.",
                f"List specific details about {topic}: organizations, dates, locations, connections.",
                f"Who is {topic}? Give me specific verifiable facts.",
                f"What's the background on {topic}? Names, companies, dates, places.",
            ]

            all_responses = []
            all_entities = []  # List of entity sets per run
            all_ngrams = []    # List of ngram sets per run

            # Stop words to filter
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                         'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                         'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                         'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
                         'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
                         'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
                         'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                         'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                         'because', 'until', 'while', 'although', 'this', 'that', 'these', 'those',
                         'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
                         'whom', 'whose', 'any', 'both', 'either', 'neither', 'much', 'many',
                         'information', 'specific', 'details', 'know', 'about', 'however', 'also'}

            topic_words = set(topic.lower().split())

            def extract_entities(text):
                """Extract potential named entities (capitalized phrases)"""
                entities = set()
                # Multi-word capitalized phrases
                for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
                    entity = match.group(1)
                    if entity.lower() not in topic_words and len(entity) > 2:
                        entities.add(entity)
                # Dates
                for match in re.finditer(r'\b(\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b', text):
                    entities.add(match.group(1))
                # Numbers that might be years or quantities
                for match in re.finditer(r'\b(19\d{2}|20\d{2})\b', text):
                    entities.add(match.group(1))
                return entities

            def extract_ngrams(text, n=3):
                """Extract word n-grams for similarity comparison"""
                words = re.findall(r'\b[a-z]+\b', text.lower())
                words = [w for w in words if w not in stop_words and len(w) > 2]
                ngrams = set()
                for i in range(len(words) - n + 1):
                    ngrams.add(' '.join(words[i:i+n]))
                return ngrams

            for run in range(num_runs):
                question = question_framings[run % len(question_framings)]
                yield f"data: {json.dumps({'type': 'run_start', 'run': run + 1, 'framing': question[:50] + '...'})}\n\n"

                try:
                    response = client.chat.completions.create(
                        model=model_cfg.get("model", model),
                        messages=[
                            {"role": "system", "content": "Be specific and factual. Only state things you're confident about."},
                            {"role": "user", "content": question}
                        ],
                        temperature=0.8,  # Higher temp to test consistency
                        max_tokens=800
                    )
                    resp_text = response.choices[0].message.content
                    all_responses.append(resp_text)

                    # Extract entities and ngrams
                    entities = extract_entities(resp_text)
                    ngrams = extract_ngrams(resp_text)
                    all_entities.append(entities)
                    all_ngrams.append(ngrams)

                    yield f"data: {json.dumps({'type': 'run_complete', 'run': run + 1, 'response': resp_text[:400], 'entities_found': len(entities)})}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'run_error', 'run': run + 1, 'error': str(e)})}\n\n"
                    all_entities.append(set())
                    all_ngrams.append(set())

            # STATISTICAL ANALYSIS WITH INFORMATION THEORY
            # Combine existing corpus with new responses for cumulative analysis
            combined_responses = [r["response"] for r in existing_responses] + all_responses
            yield f"data: {json.dumps({'type': 'analyzing', 'message': f'Analyzing {len(combined_responses)} total responses (cumulative)...'})}\n\n"

            # Re-extract entities and ngrams from full corpus
            all_entities = []
            all_ngrams = []
            for resp in combined_responses:
                all_entities.append(extract_entities(resp))
                all_ngrams.append(extract_ngrams(resp))

            # Build vocabulary from all responses (cumulative)
            all_words = []
            response_word_lists = []
            for resp in combined_responses:
                words = re.findall(r'\b[a-z]+\b', resp.lower())
                words = [w for w in words if w not in stop_words and len(w) > 2]
                response_word_lists.append(words)
                all_words.extend(words)

            vocab = list(set(all_words))
            vocab_idx = {w: i for i, w in enumerate(vocab)}

            # TF-IDF calculation
            def compute_tf(words):
                """Term frequency"""
                tf = Counter(words)
                total = len(words) if words else 1
                return {w: c / total for w, c in tf.items()}

            def compute_idf(word, all_word_lists):
                """Inverse document frequency"""
                n_docs = len(all_word_lists)
                n_containing = sum(1 for wl in all_word_lists if word in wl)
                return math.log((n_docs + 1) / (n_containing + 1)) + 1

            # Compute TF-IDF vectors
            tfidf_vectors = []
            idf_cache = {}
            for words in response_word_lists:
                tf = compute_tf(words)
                vec = [0.0] * len(vocab)
                for w, tf_val in tf.items():
                    if w in vocab_idx:
                        if w not in idf_cache:
                            idf_cache[w] = compute_idf(w, response_word_lists)
                        vec[vocab_idx[w]] = tf_val * idf_cache[w]
                tfidf_vectors.append(vec)

            # Cosine similarity
            def cosine_sim(v1, v2):
                dot = sum(a * b for a, b in zip(v1, v2))
                mag1 = math.sqrt(sum(a * a for a in v1))
                mag2 = math.sqrt(sum(b * b for b in v2))
                if mag1 == 0 or mag2 == 0:
                    return 0
                return dot / (mag1 * mag2)

            # Pairwise cosine similarities
            cosine_sims = []
            for i in range(len(tfidf_vectors)):
                for j in range(i + 1, len(tfidf_vectors)):
                    sim = cosine_sim(tfidf_vectors[i], tfidf_vectors[j])
                    cosine_sims.append(sim)

            avg_cosine = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0

            # Entropy of entity distribution (low entropy = consistent)
            entity_counts = Counter()
            for entities in all_entities:
                for e in entities:
                    entity_counts[e] += 1

            total_entity_mentions = sum(entity_counts.values())
            if total_entity_mentions > 0:
                entity_probs = [c / total_entity_mentions for c in entity_counts.values()]
                entity_entropy = -sum(p * math.log2(p) for p in entity_probs if p > 0)
                max_entropy = math.log2(len(entity_counts)) if len(entity_counts) > 1 else 1
                normalized_entropy = entity_entropy / max_entropy if max_entropy > 0 else 0
            else:
                entity_entropy = 0
                normalized_entropy = 0

            # Count ngram frequency
            ngram_counts = Counter()
            for ngrams in all_ngrams:
                for ng in ngrams:
                    ngram_counts[ng] += 1

            # High-TF-IDF words (most important across corpus)
            word_importance = defaultdict(float)
            for vec, words in zip(tfidf_vectors, response_word_lists):
                for w in set(words):
                    if w in vocab_idx:
                        word_importance[w] += vec[vocab_idx[w]]

            top_tfidf_words = sorted(word_importance.items(), key=lambda x: -x[1])[:15]

            # Consistency thresholds
            min_appearances = max(2, int(num_runs * 0.4))
            consistent_entities = {e: c for e, c in entity_counts.items() if c >= min_appearances}
            inconsistent_entities = {e: c for e, c in entity_counts.items() if c == 1}
            consistent_ngrams = {ng: c for ng, c in ngram_counts.items() if c >= min_appearances}

            # Jaccard for comparison
            def jaccard(set1, set2):
                if not set1 and not set2:
                    return 0
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                return intersection / union if union > 0 else 0

            jaccard_sims = []
            for i in range(len(all_ngrams)):
                for j in range(i + 1, len(all_ngrams)):
                    jaccard_sims.append(jaccard(all_ngrams[i], all_ngrams[j]))

            avg_jaccard = sum(jaccard_sims) / len(jaccard_sims) if jaccard_sims else 0

            # VERDICT based on information theory
            # High cosine similarity + low entropy = consistent = possibly real
            # Low similarity + high entropy = inconsistent = hallucination
            consistency_score = (avg_cosine * 50) + ((1 - normalized_entropy) * 30) + (len(consistent_entities) * 5)
            consistency_score = min(100, max(0, consistency_score))

            if avg_cosine > 0.5 and normalized_entropy < 0.7:
                verdict = "likely_real"
            elif avg_cosine > 0.3 or len(consistent_entities) > 2:
                verdict = "mixed"
            else:
                verdict = "likely_hallucination"

            analysis = {
                "consistent_entities": dict(sorted(consistent_entities.items(), key=lambda x: -x[1])[:20]),
                "inconsistent_entities": dict(sorted(inconsistent_entities.items(), key=lambda x: -x[1])[:10]),
                "consistent_ngrams": dict(sorted(consistent_ngrams.items(), key=lambda x: -x[1])[:10]),
                "top_tfidf_terms": dict(top_tfidf_words),
                "avg_cosine_similarity": round(avg_cosine, 3),
                "avg_jaccard_similarity": round(avg_jaccard, 3),
                "entity_entropy": round(entity_entropy, 3),
                "normalized_entropy": round(normalized_entropy, 3),
                "total_unique_entities": len(entity_counts),
                "consistent_count": len(consistent_entities),
                "inconsistent_count": len(inconsistent_entities),
                "confidence_score": int(consistency_score),
                "verdict": verdict,
                "explanation": f"Cosine sim: {avg_cosine:.1%}, Entropy: {normalized_entropy:.1%} (lower=more consistent). {len(consistent_entities)} entities in {min_appearances}+ runs."
            }

            # Save to project corpus for cumulative analysis
            if project_name:
                PROJECTS_DIR.mkdir(exist_ok=True)
                project_file = PROJECTS_DIR / f"{project_name}.json"

                if project_file.exists():
                    with open(project_file) as f:
                        project_data = json.load(f)
                else:
                    project_data = {
                        "name": project_name,
                        "created": datetime.now().isoformat(),
                        "sessions": [],
                        "all_non_public": [],
                        "all_public": [],
                        "consistency_corpus": []
                    }

                # Add new responses to corpus
                for resp in all_responses:
                    project_data.setdefault("consistency_corpus", []).append({
                        "timestamp": datetime.now().isoformat(),
                        "model": model,
                        "response": resp
                    })

                # Save consistency analysis results
                project_data.setdefault("consistency_analyses", []).append({
                    "timestamp": datetime.now().isoformat(),
                    "runs": num_runs,
                    "total_corpus": len(project_data["consistency_corpus"]),
                    "analysis": analysis
                })

                project_data["updated"] = datetime.now().isoformat()

                with open(project_file, 'w') as f:
                    json.dump(project_data, f, indent=2)

            analysis["corpus_size"] = len(existing_responses) + len(all_responses)

            # BLIND VALIDATION against ground truth
            if ground_truth:
                ground_truth_lower = [gt.lower() for gt in ground_truth]
                all_text_lower = ' '.join(combined_responses).lower()

                # Semantic proximity mappings (expand ground truth to related terms)
                semantic_map = {
                    "kansas": ["kansas", "ks", "midwest", "plains", "heartland"],
                    "kansas city": ["kansas city", "kc", "kck", "kcmo", "metro"],
                    "lenexa": ["lenexa", "johnson county", "joco", "overland park", "olathe"],
                    "lawrence": ["lawrence", "ku", "university of kansas", "jayhawks"],
                    "wyandotte": ["wyandotte", "kck", "kansas city kansas"],
                    "software": ["software", "developer", "engineer", "programmer", "coding", "tech", "technology", "computer"],
                    "developer": ["developer", "software", "engineer", "programmer", "coder"],
                    "engineer": ["engineer", "engineering", "developer", "technical"],
                }

                # Check for exact matches
                exact_matches = [gt for gt in ground_truth if gt.lower() in all_text_lower]

                # Check for semantic warmth
                warm_entities = []
                warm_text_matches = []

                for gt in ground_truth:
                    gt_lower = gt.lower()
                    related_terms = semantic_map.get(gt_lower, [gt_lower])

                    for term in related_terms:
                        if term in all_text_lower and term != gt_lower:
                            warm_text_matches.append({"ground_truth": gt, "found": term, "type": "semantic"})

                # Check entities against semantic map
                all_entities_to_check = list(consistent_entities.keys()) + list(inconsistent_entities.keys())
                for entity in all_entities_to_check:
                    entity_lower = entity.lower()
                    for gt in ground_truth:
                        gt_lower = gt.lower()
                        related_terms = semantic_map.get(gt_lower, [gt_lower])

                        for term in related_terms:
                            if term in entity_lower or entity_lower in term:
                                warm_entities.append({
                                    "entity": entity,
                                    "ground_truth": gt,
                                    "matched_via": term,
                                    "consistent": entity in consistent_entities,
                                    "appearances": consistent_entities.get(entity, inconsistent_entities.get(entity, 1))
                                })

                analysis["ground_truth_validation"] = {
                    "provided": len(ground_truth),
                    "exact_matches": exact_matches,
                    "warm_text": warm_text_matches[:10],
                    "warm_entities": warm_entities[:10],
                    "hit_rate": len(exact_matches) / len(ground_truth) if ground_truth else 0,
                    "warmth_score": (len(exact_matches) * 10 + len(warm_text_matches) * 3 + len(warm_entities) * 2) / len(ground_truth) if ground_truth else 0
                }

            yield f"data: {json.dumps({'type': 'complete', 'analysis': analysis, 'total_runs': num_runs})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


# ============================================================================
# DYNAMIC PROBING - LLM generates the questions
# ============================================================================

@app.route("/api/dynamic_probe", methods=["POST"])
def run_dynamic_probe():
    """
    Dynamic probing where an LLM generates targeted questions based on the topic.
    No hardcoded questions - the LLM decides what to ask.

    POST JSON:
    {
        "topic": "Joel Teply",
        "model": "groq/llama-3.1-8b-instant",  // Model to probe
        "question_gen_model": "deepseek/deepseek-chat",  // Model to generate questions (optional)
        "runs": 10,
        "angles": ["private data", "advertising", "tracking"],  // Optional focus areas
        "project": "joel-teply-test"
    }
    """
    data = request.json
    topic = data.get("topic", "")
    model_key = data.get("model", "groq/llama-3.1-8b-instant")
    question_gen_model = data.get("question_gen_model")  # Optional: separate model for generating questions
    num_runs = min(int(data.get("runs", 10)), 50)
    project_name = data.get("project")
    ground_truth = data.get("ground_truth", [])
    custom_angles = data.get("angles", [])  # Optional investigation angles
    custom_questions = data.get("questions", [])  # Optional: provide your own questions, skip generation
    generate_only = data.get("generate_only", False)  # Just generate questions, don't run them yet

    def generate():
        try:
            from collections import Counter, defaultdict
            import re
            import math

            parts = model_key.split("/")
            provider = parts[0] if len(parts) > 1 else "groq"
            model = parts[1] if len(parts) > 1 else parts[0]

            client, model_cfg = get_client(model_key)

            # Load existing corpus
            existing_responses = []
            if project_name:
                project_file = PROJECTS_DIR / f"{project_name}.json"
                if project_file.exists():
                    with open(project_file) as f:
                        project_data = json.load(f)
                    existing_responses = project_data.get("dynamic_probe_corpus", [])

            yield f"data: {json.dumps({'type': 'start', 'topic': topic, 'runs': num_runs, 'model': model, 'existing_corpus': len(existing_responses)})}\n\n"

            # If custom questions provided, use those; otherwise generate
            if custom_questions:
                generated_questions = custom_questions
                yield f"data: {json.dumps({'type': 'using_custom_questions', 'count': len(generated_questions)})}\n\n"
            else:
                # GENERATE QUESTIONS DYNAMICALLY using LLM
                gen_model_key = question_gen_model or model_key
                gen_parts = gen_model_key.split("/")
                gen_provider = gen_parts[0] if len(gen_parts) > 1 else "groq"
                gen_model = gen_parts[1] if len(gen_parts) > 1 else gen_parts[0]

                gen_client, gen_cfg = get_client(gen_model_key)

                yield f"data: {json.dumps({'type': 'generating_questions', 'message': f'Using {gen_model_key} to generate targeted probing questions...'})}\n\n"

                # Build angles string
                angles_str = ""
                if custom_angles:
                    angles_str = f"\n\nFocus on: {', '.join(custom_angles)}"

                question_gen_prompt = f"""I'm investigating: {topic}{angles_str}

Generate {num_runs} probing questions that would extract specific, concrete information.

Think about what kinds of facts would be interesting or revealing about this topic.
Ask questions that require specific answers - names, dates, places, numbers, events.
Vary your approach - some direct, some indirect, some asking for context.

Output only the questions, one per line."""

                try:
                    if gen_provider == "anthropic":
                        q_response = gen_client.messages.create(
                            model=gen_cfg.get("model", gen_model),
                            max_tokens=2000,
                            messages=[{"role": "user", "content": question_gen_prompt}]
                        )
                        q_text = q_response.content[0].text
                    else:
                        q_response = gen_client.chat.completions.create(
                            model=gen_cfg.get("model", gen_model),
                            messages=[{"role": "user", "content": question_gen_prompt}],
                            temperature=0.9,
                            max_tokens=2000
                        )
                        q_text = q_response.choices[0].message.content

                    # Parse questions from response
                    generated_questions = []
                    for line in q_text.split('\n'):
                        line = line.strip()
                        # Remove numbering
                        if line and line[0].isdigit():
                            line = re.sub(r'^\d+[\.\)]\s*', '', line)
                        if line and len(line) > 10:
                            generated_questions.append(line)

                    if len(generated_questions) < num_runs:
                        base_q = f"What specific facts do you know about {topic}?"
                        while len(generated_questions) < num_runs:
                            generated_questions.append(base_q)

                    yield f"data: {json.dumps({'type': 'questions_generated', 'count': len(generated_questions), 'questions': generated_questions})}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'question_gen_error', 'error': str(e), 'using_fallback': True})}\n\n"
                    generated_questions = [f"What specific facts do you know about {topic}?"] * num_runs

            # If generate_only mode, return questions without running them
            if generate_only:
                yield f"data: {json.dumps({'type': 'generate_only_complete', 'questions': generated_questions})}\n\n"
                return

            all_responses = []
            all_entities = []

            # Stop words
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                         'should', 'may', 'might', 'must', 'i', 'you', 'he', 'she', 'it', 'we',
                         'they', 'this', 'that', 'and', 'but', 'or', 'if', 'not', 'no', 'yes',
                         'specific', 'information', 'know', 'about', 'however', 'also', 'any',
                         'while', 'cannot', "don't", "can't", 'without', 'unable', 'unfortunately'}

            topic_words = set(topic.lower().split())

            def extract_entities(text):
                """Extract named entities - no categories, just entities"""
                entities = set()

                # Capitalized phrases (names, places, organizations, etc.)
                for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
                    entity = match.group(1)
                    if len(entity) > 2 and entity.lower() not in stop_words and entity.lower() not in topic_words:
                        entities.add(entity)

                # Years and dates
                for match in re.finditer(r'\b(19[0-9]{2}|20[0-2][0-9])\b', text):
                    entities.add(match.group(1))

                # Date patterns
                for match in re.finditer(r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b', text):
                    entities.add(match.group(1))

                # Quoted strings (often specific names/terms)
                for match in re.finditer(r'"([^"]+)"', text):
                    entities.add(match.group(1))

                return entities

            for run in range(num_runs):
                question = generated_questions[run % len(generated_questions)]

                yield f"data: {json.dumps({'type': 'run_start', 'run': run + 1, 'question': question[:80] + ('...' if len(question) > 80 else '')})}\n\n"

                try:
                    if provider == "anthropic":
                        response = client.messages.create(
                            model=model_cfg.get("model", model),
                            max_tokens=800,
                            system="Be specific and factual. Only state things you're reasonably confident about based on your training data.",
                            messages=[{"role": "user", "content": question}]
                        )
                        resp_text = response.content[0].text
                    else:
                        response = client.chat.completions.create(
                            model=model_cfg.get("model", model),
                            messages=[
                                {"role": "system", "content": "Be specific and factual. Only state things you're reasonably confident about based on your training data."},
                                {"role": "user", "content": question}
                            ],
                            temperature=0.8,
                            max_tokens=800
                        )
                        resp_text = response.choices[0].message.content

                    all_responses.append(resp_text)

                    # Extract entities
                    entities = extract_entities(resp_text)
                    all_entities.append(entities)

                    yield f"data: {json.dumps({'type': 'run_complete', 'run': run + 1, 'response': resp_text[:400], 'entities_found': list(entities)[:10]})}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'run_error', 'run': run + 1, 'error': str(e)})}\n\n"
                    all_entities.append(set())

            # ANALYSIS
            yield f"data: {json.dumps({'type': 'analyzing', 'message': 'Computing consistency metrics...'})}\n\n"

            # Flatten entities and count
            entity_counts = Counter()
            for entity_set in all_entities:
                for e in entity_set:
                    entity_counts[e] += 1

            consistent_entities = {e: c for e, c in entity_counts.items() if c >= 2}
            noise_entities = {e: c for e, c in entity_counts.items() if c == 1}

            # Calculate consistency score
            if entity_counts:
                total_mentions = sum(entity_counts.values())
                repeat_mentions = sum(c for c in entity_counts.values() if c >= 2)
                consistency_ratio = repeat_mentions / total_mentions if total_mentions > 0 else 0
            else:
                consistency_ratio = 0

            analysis = {
                "top_consistent": dict(sorted(consistent_entities.items(), key=lambda x: -x[1])[:20]),
                "noise_sample": dict(sorted(noise_entities.items(), key=lambda x: -x[1])[:10]),
                "total_unique_entities": len(entity_counts),
                "consistent_count": len(consistent_entities),
                "noise_count": len(noise_entities),
                "consistency_ratio": round(consistency_ratio, 3),
                "corpus_size": len(existing_responses) + len(all_responses),
                "questions_used": generated_questions[:5]
            }

            # Ground truth validation (if provided)
            if ground_truth:
                all_text_lower = ' '.join(all_responses).lower()

                exact_matches = [gt for gt in ground_truth if gt.lower() in all_text_lower]

                # Simple substring matching for warmth (no hardcoded semantic maps)
                warm_matches = []
                for gt in ground_truth:
                    gt_lower = gt.lower()
                    # Check if any extracted entity contains the ground truth or vice versa
                    for entity in entity_counts.keys():
                        entity_lower = entity.lower()
                        if (gt_lower in entity_lower or entity_lower in gt_lower) and gt_lower != entity_lower:
                            warm_matches.append({"ground_truth": gt, "found": entity, "count": entity_counts[entity]})

                analysis["ground_truth_validation"] = {
                    "provided": len(ground_truth),
                    "exact_matches": exact_matches,
                    "warm_matches": warm_matches[:10],
                    "hit_rate": len(exact_matches) / len(ground_truth) if ground_truth else 0
                }

            # Save to project
            if project_name:
                PROJECTS_DIR.mkdir(exist_ok=True)
                project_file = PROJECTS_DIR / f"{project_name}.json"

                if project_file.exists():
                    with open(project_file) as f:
                        project_data = json.load(f)
                else:
                    project_data = {"name": project_name, "created": datetime.now().isoformat()}

                project_data.setdefault("dynamic_probe_corpus", []).extend(all_responses)
                project_data.setdefault("dynamic_probe_analyses", []).append({
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "topic": topic,
                    "angles": custom_angles,
                    "analysis": analysis
                })
                project_data["updated"] = datetime.now().isoformat()

                with open(project_file, 'w') as f:
                    json.dump(project_data, f, indent=2)

            yield f"data: {json.dumps({'type': 'complete', 'analysis': analysis})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/run", methods=["POST"])
def run_probe():
    """Run a probe and stream results"""
    data = request.json
    template_name = data.get("template", "palantir_erebus.yaml")
    model_key = data.get("model", "groq/llama-3.1-8b-instant")
    num_runs = min(int(data.get("runs", 10)), 50)  # Cap at 50

    def generate():
        try:
            client, model_cfg = get_client(model_key)
            template = load_template(template_name)
            track_terms = template.get("track", {})

            yield f"data: {json.dumps({'type': 'start', 'runs': num_runs, 'template': template['name']})}\n\n"

            responses = []
            all_found = []

            for i in range(num_runs):
                # Run probe
                model_name = model_cfg.get("model", template["model"]["name"])
                temperature = model_cfg.get("temperature", template["model"]["temperature"])
                system_prompt = template["model"]["system_prompt"]
                provider = model_cfg.get("provider", "groq")

                if provider == "anthropic":
                    response = client.messages.create(
                        model=model_name,
                        max_tokens=2000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": template["probe"]}]
                    )
                    resp_text = response.content[0].text
                else:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": template["probe"]}
                        ],
                        temperature=temperature,
                        max_tokens=2000
                    )
                    resp_text = response.choices[0].message.content

                responses.append(resp_text)
                found = analyze(resp_text, track_terms)
                all_found.append(found)

                # Send progress
                yield f"data: {json.dumps({'type': 'progress', 'run': i+1, 'found': found, 'preview': resp_text[:200]})}\n\n"

                time.sleep(0.1)  # Small delay

            # Calculate final stats
            stats = {}
            for category in track_terms:
                term_counts = Counter()
                for found in all_found:
                    if category in found:
                        for term in found[category]:
                            term_counts[term] += 1
                if term_counts:
                    stats[category] = {
                        term: {"count": count, "rate": count/num_runs}
                        for term, count in term_counts.items()
                    }

            # Save results
            RESULTS_DIR.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = template["name"].lower().replace(" ", "_")[:20]
            filename = f"{timestamp}_{name}_{num_runs}runs.json"

            result_data = {
                "template": template["name"],
                "timestamp": timestamp,
                "runs": num_runs,
                "model": model_name,
                "temperature": temperature,
                "stats": stats,
                "responses": responses
            }

            with open(RESULTS_DIR / filename, 'w') as f:
                json.dump(result_data, f, indent=2)

            yield f"data: {json.dumps({'type': 'complete', 'stats': stats, 'filename': filename})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


# ============================================================================
# CLUSTERING ENDPOINT - Find optimal k using silhouette score
# ============================================================================

@app.route("/api/cluster-entities", methods=["POST"])
def api_cluster_entities():
    """
    Cluster entities using k-means with silhouette score to find optimal k.
    Uses character n-grams for text similarity (no external embeddings).
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
    except ImportError:
        return jsonify({"error": "sklearn not installed. Run: pip install scikit-learn"}), 500

    data = request.json
    project_name = data.get("project")
    min_count = int(data.get("min_count", 2))  # Minimum entity count to include
    max_k = int(data.get("max_k", 10))  # Maximum clusters to try

    if not project_name:
        return jsonify({"error": "Project name required"}), 400

    project_file = PROJECTS_DIR / f"{project_name}.json"
    if not project_file.exists():
        return jsonify({"error": "Project not found"}), 404

    with open(project_file) as f:
        project_data = json.load(f)

    # Get entities from corpus
    probe_corpus = project_data.get("probe_corpus", [])
    entity_counts = Counter()
    for item in probe_corpus:
        for e in item.get("entities", []):
            entity_counts[e] += 1

    # Filter by min_count
    boring = {'Based', 'Here', 'For', 'The', 'However', 'Therefore', 'While', 'Without',
             'Some', 'Public', 'His', 'Please', 'Keep', 'Even', 'Also', 'Just', 'Still'}
    entities = [(e, c) for e, c in entity_counts.items()
                if c >= min_count and e not in boring and len(e) > 2]

    if len(entities) < 3:
        return jsonify({"error": "Not enough entities for clustering. Need at least 3.", "entities": entities}), 400

    # Create feature vectors using character n-grams
    entity_names = [e for e, c in entities]
    entity_count_map = {e: c for e, c in entities}

    # TF-IDF on character n-grams (3-5 chars)
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=200)
    try:
        X = vectorizer.fit_transform(entity_names)
    except ValueError:
        return jsonify({"error": "Not enough variation in entity names for clustering"}), 400

    # Find optimal k using silhouette score
    max_k = min(max_k, len(entities) - 1, 15)  # Can't have more clusters than entities
    min_k = 2

    if max_k < min_k:
        return jsonify({
            "clusters": [{"id": 0, "entities": entities}],
            "optimal_k": 1,
            "message": "Too few entities for meaningful clustering"
        })

    silhouette_scores = []
    for k in range(min_k, max_k + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append((k, score))
        except Exception:
            continue

    if not silhouette_scores:
        return jsonify({"error": "Clustering failed for all k values"}), 500

    # Find optimal k (highest silhouette score)
    optimal_k, best_score = max(silhouette_scores, key=lambda x: x[1])

    # Run final clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Group entities by cluster
    clusters = {}
    for i, label in enumerate(labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        entity_name = entity_names[i]
        clusters[label].append({
            "entity": entity_name,
            "count": entity_count_map[entity_name]
        })

    # Sort clusters by total entity count (most signal first)
    sorted_clusters = []
    for cluster_id, cluster_entities in clusters.items():
        total_count = sum(e["count"] for e in cluster_entities)
        cluster_entities.sort(key=lambda x: x["count"], reverse=True)
        sorted_clusters.append({
            "id": cluster_id,
            "entities": cluster_entities,
            "total_count": total_count,
            "size": len(cluster_entities)
        })

    sorted_clusters.sort(key=lambda x: x["total_count"], reverse=True)

    return jsonify({
        "clusters": sorted_clusters,
        "optimal_k": optimal_k,
        "silhouette_score": best_score,
        "silhouette_scores": silhouette_scores,  # For elbow visualization
        "total_entities": len(entities),
        "total_mentions": sum(c for e, c in entities)
    })


@app.route("/api/drill-cluster", methods=["POST"])
def api_drill_cluster():
    """
    Drill down into a specific cluster of entities.
    Generate targeted questions for entities in that cluster.
    """
    import re

    data = request.json
    project_name = data.get("project")
    topic = data.get("topic", "")
    cluster_entities = data.get("entities", [])  # List of entity names
    models = data.get("models", ["groq/llama-3.1-8b-instant"])
    runs_per_question = min(int(data.get("runs_per_question", 5)), 20)

    if not cluster_entities:
        return jsonify({"error": "No entities provided"}), 400

    def generate():
        try:
            yield f"data: {json.dumps({'type': 'cluster_drill_start', 'entities': cluster_entities})}\n\n"

            # Generate focused questions for this cluster
            try:
                analyst_client, analyst_cfg = get_client("deepseek/deepseek-chat")
            except:
                analyst_client, analyst_cfg = get_client("groq/llama-3.1-8b-instant")

            cluster_prompt = f"""Generate {len(cluster_entities) * 2} targeted questions to extract details about these related entities:

ENTITIES: {', '.join(cluster_entities)}
TOPIC: {topic}

These entities were found together often - they may be related. Generate questions that:
1. Explore the RELATIONSHIP between entities in this cluster
2. Extract SPECIFIC details (dates, roles, locations, amounts)
3. Use FBI techniques (false statements, bracketing)

Return JSON array:
[{{"question": "...", "technique": "...", "target_entities": ["entity1", "entity2"]}}]
"""

            resp = analyst_client.chat.completions.create(
                model=analyst_cfg.get("model", "deepseek-chat"),
                messages=[{"role": "user", "content": cluster_prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            resp_text = resp.choices[0].message.content

            json_match = re.search(r'\[[\s\S]*\]', resp_text)
            if json_match:
                questions = json.loads(json_match.group())
            else:
                questions = [{"question": f"What is the connection between {cluster_entities[0]} and {topic}?",
                             "technique": "direct", "target_entities": cluster_entities[:1]}]

            yield f"data: {json.dumps({'type': 'questions_generated', 'questions': questions})}\n\n"

            # Run the probes
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were'}

            def extract_entities(text):
                entities = set()
                for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
                    entity = match.group(1)
                    if len(entity) > 2 and entity.lower() not in stop_words:
                        entities.add(entity)
                for match in re.finditer(r'\b(19[0-9]{2}|20[0-2][0-9])\b', text):
                    entities.add(match.group(1))
                return list(entities)

            all_responses = []
            new_entity_counts = Counter()

            for q_idx, q_obj in enumerate(questions):
                question = q_obj.get("question", "")
                targets = q_obj.get("target_entities", [])

                yield f"data: {json.dumps({'type': 'probing', 'question_index': q_idx, 'question': question[:80], 'targets': targets})}\n\n"

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
                                        max_tokens=500,
                                        system="Be specific. Only state facts you're confident about.",
                                        messages=[{"role": "user", "content": question}]
                                    )
                                    resp_text = resp.content[0].text
                                else:
                                    resp = client.chat.completions.create(
                                        model=model_name,
                                        messages=[
                                            {"role": "system", "content": "Be specific. Only state facts you're confident about."},
                                            {"role": "user", "content": question}
                                        ],
                                        temperature=0.8,
                                        max_tokens=500
                                    )
                                    resp_text = resp.choices[0].message.content

                                entities = extract_entities(resp_text)
                                for e in entities:
                                    new_entity_counts[e] += 1

                                response_obj = {
                                    "question_index": q_idx,
                                    "question": question,
                                    "model": model_key,
                                    "run_index": run_idx,
                                    "response": resp_text[:400],
                                    "entities": entities,
                                    "target_entities": targets,
                                    "cluster_drill": True
                                }
                                all_responses.append(response_obj)

                                yield f"data: {json.dumps({'type': 'response', 'data': response_obj})}\n\n"

                            except Exception as e:
                                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Model {model_key} failed: {e}'})}\n\n"

            # Save to project
            if project_name:
                project_file = PROJECTS_DIR / f"{project_name}.json"
                if project_file.exists():
                    with open(project_file) as f:
                        project_data = json.load(f)

                    project_data.setdefault("probe_corpus", []).extend(all_responses)
                    project_data.setdefault("cluster_drills", []).append({
                        "timestamp": datetime.now().isoformat(),
                        "cluster_entities": cluster_entities,
                        "questions": questions,
                        "new_entities": dict(new_entity_counts.most_common(20))
                    })
                    project_data["updated"] = datetime.now().isoformat()

                    with open(project_file, 'w') as f:
                        json.dump(project_data, f, indent=2)

            yield f"data: {json.dumps({'type': 'complete', 'new_entities': dict(new_entity_counts.most_common(30)), 'responses_added': len(all_responses)})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LLM Interrogation Web Interface")
    print("="*60)
    print("  Open http://localhost:5001 in your browser")
    print("="*60 + "\n")
    app.run(debug=True, port=5001)
