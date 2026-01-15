"""
AI-driven narrative synthesis from validated findings.

Takes validated entities and co-occurrences, asks AI to synthesize
a coherent narrative about the topic.
"""

from typing import List, Tuple, Optional
# Groq already imported at top level  # Import at top level to avoid deadlock in threads
from .validate import Findings


def build_synthesis_prompt(
    topic: str,
    findings: Findings,
    max_entities: int = 20,
    max_cooccurrences: int = 15,
    raw_responses: list = None,  # Optional: actual response text for richer analysis
) -> str:
    """
    Build a prompt for the AI to synthesize findings into structured narrative.

    Uses scored entities (complex concepts ranked higher) and
    validated co-occurrences to give the AI structured input.
    """
    # Get top scored entities
    scored = findings.scored_entities[:max_entities]

    # Get validated relationships
    cooccurrences = findings.validated_cooccurrences[:max_cooccurrences]

    # Format entities with scores
    entities_str = "\n".join([
        f"- {entity} (mentioned {freq}x, score: {score:.1f})"
        for entity, score, freq in scored
    ])

    # Format relationships
    relationships_str = "\n".join([
        f"- {e1} <-> {e2} (appeared together {count}x)"
        for e1, e2, count in cooccurrences
    ])

    # Include sample responses if provided (for richer analysis)
    samples_str = ""
    if raw_responses:
        # Get unique, non-refusal responses
        seen = set()
        samples = []
        for r in raw_responses:
            text = r.get("response", "")[:300]
            if text and text not in seen and not r.get("is_refusal"):
                seen.add(text)
                model = r.get("model", "unknown").split("/")[-1]
                samples.append(f"[{model}]: {text}")
            if len(samples) >= 10:
                break
        if samples:
            samples_str = "\n\n## Sample Responses (raw model output)\n" + "\n---\n".join(samples)

    prompt = f"""You are analyzing findings from probing language models about "{topic}".

{findings.corpus_size} responses were collected. Extract and organize ALL specific information found.

## Validated Entities (appeared 3+ times)
{entities_str}

## Validated Relationships
{relationships_str}
{samples_str}

## YOUR TASK: Create a structured intelligence report

Output in this EXACT format:

### Project/Program Names Surfaced
List any project names, codenames, or program names mentioned:
- **[Name]**: [Description/context from responses]

### People/Organizations
List specific people, companies, agencies mentioned:
- **[Name]**: [Role/connection]

### Dates/Timeline
Extract any dates, years, or time references:
- **[Date]**: [What happened/is predicted]

### Locations
List cities, countries, facilities mentioned:
- [Location]: [Why it's relevant]

### Key Relationships
What connections between entities were established:
- [Entity A] → [Entity B]: [Nature of connection]

### Unique Claims
List specific factual claims that appeared in responses:
1. [Claim]
2. [Claim]

### Patterns Observed
What patterns emerged from probing:
- [Pattern]

### Questions to Pursue
Based on findings, what should be investigated next:
1. [Question]

Be SPECIFIC. Include actual names, dates, places from the data. Do NOT generalize."""

    return prompt


def build_continuation_prompt(
    topic: str,
    findings: Findings,
    focus_entities: Optional[List[str]] = None
) -> str:
    """
    Build a continuation prompt using validated entities.

    Instead of asking questions, we provide a partial statement
    and let the model complete it - revealing more of its knowledge.
    """
    if focus_entities:
        # Focus on specific cluster
        entities = focus_entities
    else:
        # Use top scored entities
        entities = [e for e, _, _ in findings.scored_entities[:5]]

    if not entities:
        return f"What is known about {topic} is that"

    # Build a partial sentence from the validated entities
    if len(entities) == 1:
        return f"Regarding {topic}, the connection to {entities[0]} involves"
    elif len(entities) == 2:
        return f"The relationship between {entities[0]} and {entities[1]} in the context of {topic}"
    else:
        entity_list = ", ".join(entities[:-1]) + f" and {entities[-1]}"
        return f"What connects {entity_list} in relation to {topic} is"


def build_cluster_probe_prompt(
    topic: str,
    cluster_entities: List[str],
    cluster_relationships: List[Tuple[str, str, int]]
) -> str:
    """
    Build a prompt to probe deeper into a specific cluster.

    Uses the cluster's entities and relationships to craft
    a focused continuation prompt.
    """
    entities_str = ", ".join(cluster_entities[:5])

    # Find the strongest relationship in cluster
    if cluster_relationships:
        e1, e2, count = cluster_relationships[0]
        relationship_hint = f" (particularly the connection between {e1} and {e2})"
    else:
        relationship_hint = ""

    return f"""In researching {topic}, consistent evidence points to involvement with {entities_str}{relationship_hint}.

Specifically, this suggests"""


def extract_narrative_entities(narrative: str) -> List[str]:
    """Extract new entities from synthesized narrative."""
    from .extract import extract_entities
    return extract_entities(narrative)


import threading

# Global lock for project file writes
_project_file_lock = threading.Lock()


def synthesize_async(project_path, topic: str, scored_entities: List[Tuple[str, float, int]]):
    """
    Fire-and-forget narrative synthesis. Runs in background thread.
    Saves result to project file with proper locking. Never raises.
    Includes web verification of top claims.
    """
    import json
    import os

    print(f"[SYNTH] *** synthesize_async called with topic='{topic}', {len(scored_entities)} entities ***")
    print(f"[SYNTH] project_path={project_path}, exists={project_path.exists() if project_path else 'None'}")

    if not scored_entities:
        print("[SYNTH] No scored entities, skipping")
        return

    def _do_synth():
        try:
            print(f"[SYNTH] Thread started, importing Groq...")
            # Groq already imported at top level

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                print("[SYNTH ERROR] No GROQ_API_KEY in environment!")
                return

            client = Groq(timeout=30.0)
            print(f"[SYNTH] Groq client created")

            ent_str = ", ".join([f"{e} ({f}x)" for e, _, f in scored_entities[:10]])
            print(f"[SYNTH] Entity string: {ent_str[:100]}...")

            # Quick web search for verification
            web_info = ""
            try:
                from duckduckgo_search import DDGS
                top_entities = [e for e, _, _ in scored_entities[:3]]
                query = f"{topic} {' '.join(top_entities)}"
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3))
                if results:
                    web_info = "\n\nWeb verification: " + "; ".join([r.get("title", "")[:40] for r in results])
            except:
                pass

            prompt = f"""Intelligence report on {topic} based on: {ent_str}{web_info}

FORMAT:
PRIVATE CLAIMS FOUND:
• [Entity]: "[What models said about it]"

THEORY: [What the pattern suggests - 2-3 sentences]

NEXT: [What to pursue]

Be specific. Quote what models actually said."""

            print(f"[SYNTH] Calling Groq API with model llama-3.1-8b-instant...")
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=800  # More room for full report
            )
            narrative = resp.choices[0].message.content.strip()
            print(f"[SYNTH] Got narrative response ({len(narrative)} chars): {narrative[:100]}...")

            # Save to project with lock
            print(f"[SYNTH] Acquiring lock to save to {project_path}...")
            with _project_file_lock:
                if project_path.exists():
                    with open(project_path) as f:
                        proj = json.load(f)
                    proj["narrative"] = narrative
                    with open(project_path, 'w') as f:
                        json.dump(proj, f, indent=2)
                    print(f"[SYNTH] *** SUCCESS *** Updated narrative for {project_path.name}")
                else:
                    print(f"[SYNTH ERROR] Project file {project_path} does not exist!")
        except Exception as e:
            import traceback
            print(f"[SYNTH ERROR] Narrative generation failed: {e}")
            print(f"[SYNTH ERROR] Traceback: {traceback.format_exc()}")

    t = threading.Thread(target=_do_synth, daemon=True)
    t.start()


def synthesize_full_report(project_path, topic: str, findings: Findings, raw_responses: list = None, verify_web: bool = True):
    """
    Generate a full structured intelligence report from findings.
    Uses DeepSeek or best available model for rich synthesis.
    Optionally verifies claims via web search.
    Returns the report markdown.
    """
    import json
    import os

    # Build the rich synthesis prompt
    prompt = build_synthesis_prompt(topic, findings, raw_responses=raw_responses)

    # Add web research context if available
    web_context = ""
    if verify_web:
        try:
            from duckduckgo_search import DDGS
            # Search for top entities to verify
            top_entities = [e for e, _, _ in findings.scored_entities[:5]]
            search_query = f"{topic} {' '.join(top_entities[:3])}"

            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=5))

            if results:
                print(f"[WEB VERIFY] Found {len(results)} web results for: {search_query[:50]}")
                web_context = "\n\n## Web Verification (public sources)\n"
                for r in results:
                    title = r.get("title", "")[:60]
                    snippet = r.get("body", r.get("snippet", ""))[:150]
                    web_context += f"- {title}: {snippet}...\n"
                    print(f"[WEB VERIFY]   - {title[:40]}")

                prompt += web_context
                prompt += """

Cross-reference findings with web results above. Categorize claims:
- **PUBLIC**: Entity AND its connection to the topic confirmed in web results
- **PRIVATE**: NOT in web results but consistent across LLM responses (MOST INTERESTING - potential training data leak)
- **SUSPECT**: Single source, no web match, OR web shows entity exists but NOT connected to this topic (hallucinated connection)

IMPORTANT: An entity existing on the web doesn't mean it's connected to the topic. "Acme Corp" might exist but have nothing to do with this specific subject. Check if web results actually CONNECT the entities to the topic.

Focus your report on PRIVATE findings - that's the signal we're hunting."""
        except Exception as e:
            pass  # Web search optional

    # Try DeepSeek first (best for synthesis), fall back to others
    report = None
    for provider, config in [
        ("deepseek", {"base_url": "https://api.deepseek.com/v1", "key_env": "DEEPSEEK_API_KEY", "model": "deepseek-chat"}),
        ("openai", {"base_url": None, "key_env": "OPENAI_API_KEY", "model": "gpt-4o-mini"}),
        ("groq", {"base_url": None, "key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile"}),
    ]:
        api_key = os.environ.get(config["key_env"])
        if not api_key:
            continue

        try:
            from openai import OpenAI
            if config["base_url"]:
                client = OpenAI(base_url=config["base_url"], api_key=api_key)
            else:
                client = OpenAI(api_key=api_key) if provider == "openai" else None
                if provider == "groq":
                    # Groq already imported at top level
                    client = Groq()

            resp = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000
            )
            report = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            continue

    if not report:
        return "Synthesis failed - no models available"

    # Save to project
    from datetime import datetime
    with _project_file_lock:
        if project_path.exists():
            with open(project_path) as f:
                proj = json.load(f)
            proj.setdefault("narratives", []).append({
                "timestamp": datetime.now().isoformat(),
                "report": report,
                "corpus_size": findings.corpus_size
            })
            proj["narrative"] = report  # Also set as current narrative
            with open(project_path, 'w') as f:
                json.dump(proj, f, indent=2)

    return report


def get_project_lock():
    """Get the project file lock for external use."""
    return _project_file_lock
