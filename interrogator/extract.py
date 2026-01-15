"""
Entity and relationship extraction from text responses.

Two modes:
1. LLM-based extraction (preferred): Use a fast LLM to extract typed entities
2. Regex fallback: Pattern matching for when LLM extraction fails
"""

import re
import json
from typing import List, Set, Tuple, Dict, Optional
from dataclasses import dataclass
from groq import Groq  # Import at top level to avoid deadlock in threads


# Global patterns that indicate refusals/hedging - NEVER treat as entities
REFUSAL_PATTERNS = [
    r"need to know",
    r"need more",
    r"need.* (full|specific)",
    r"please (provide|specify|clarify)",
    r"unclear|vague",
    r"which .* you",
    r"(many|various|multiple) (individuals?|people|persons?)",
    r"scenario (is|may be|could be)",
    r"it('s| is) (unclear|difficult|hard)",
    r"cannot (determine|identify|specify)",
    r"I (don't|do not|cannot|can't)",
    r"without (more|additional|specific)",
    r"(depends|depending) on",
    r"could (be|refer to)",
    r"may (be|refer to)",
    r"might (be|refer to)",
    r"(accurate|meaningful) (summary|response|answer)",
    r"across (various|different|many) fields",
    r"(full|specific) (name|context|details?)",
    r"time period and context",
    r"more (details?|information|context)",
    r"mundane or complex",
    r"(provide|give).* context",
    r"there are many",
    r"involved in the scenario",
    r"referring to",
    r"will help give",
    r"help.*give.*accurate",
    # Generic advice / not actual facts
    r"is a (source|platform|directory|website|database)",
    r"(may|might|could) (mention|contain|have|include)",
    r"(press releases|news articles|legal|compliance).*(may|might)",
    r"professional (profile|directory|platform)",
    r"(government|state) databases",
    r"is (now )?part of",  # Generic corporate facts
    r"straddles .* and",  # Geographic trivia
    r"is home to",
    r"is a name found in",
    r"(multiple|various|many) (individuals|people|fields)",
    r"publishes reports",
    r"worked at a company",  # Too vague
    r"source (of|for) (official|court|public)",
]


def is_entity_refusal(text: str) -> bool:
    """
    Check if text is an LLM refusal/hedge that should NOT be treated as entity.

    Call this BEFORE adding any entity to counts/storage.
    """
    if not text or len(text) < 3:
        return True

    text_lower = text.lower()

    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, text_lower):
            return True

    return False


def extract_facts_fast(text: str) -> List[str]:
    """
    Fast LLM-based extraction of FACTS and CLAIMS - not just names.

    CRITICAL: Filters out LLM refusals, hedging, and generic statements.

    Extracts:
    - Specific claims and assertions
    - Technical details
    - Dates, timelines, years
    - Locations, places
    - Organizations, projects, programs
    - People mentioned
    - Quantities, amounts, measurements
    - Events, incidents
    - Codenames, designations
    - Relationships between things
    """
    # Groq already imported at top level

    extraction_prompt = """Extract SPECIFIC NAMED ENTITIES from the text below.

EXTRACT (1-5 words, PROPER NOUNS only):
- Full names of people
- Company/org names (specific, not generic)
- Project/product names
- Specific dates (month+year, quarters)

DO NOT EXTRACT:
- Generic words: "University", "Company", "Project" alone
- Job titles: "software developer", "manager"
- Common locations without org context
- Single common words

Text to analyze:
{text}

Return ONLY a JSON array of entities found IN THE TEXT ABOVE.
Example format: ["John Smith", "Acme Corp", "January 2020"]
If no specific entities found, return: []"""

    try:
        client = Groq(timeout=8.0)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": extraction_prompt.format(text=text[:1500])}],
            temperature=0,
            max_tokens=600
        )
        result = resp.choices[0].message.content.strip()
        # Find JSON array
        start = result.find('[')
        end = result.rfind(']') + 1
        if start >= 0 and end > start:
            facts = json.loads(result[start:end])
            # Filter out empty/short items AND refusals using global filter
            filtered = [f for f in facts if isinstance(f, str) and len(f) > 3 and not is_entity_refusal(f)]
            return filtered
    except Exception as e:
        print(f"[EXTRACT] Fast fact extraction failed: {e}")

    # Fallback to regex extraction
    return extract_entities_regex(text)


def extract_entities_regex(text: str) -> List[str]:
    """Regex-only fallback for entity extraction."""
    concepts = extract_concepts(text)
    concepts.sort(key=lambda c: c.specificity, reverse=True)
    seen = set()
    entities = []
    for c in concepts:
        if c.text not in seen:
            seen.add(c.text)
            entities.append(c.text)
    return entities


def extract_with_llm(text: str, client, model_name: str) -> List[Dict]:
    """
    Use LLM to extract typed entities with context.

    Returns list of {"name": str, "type": str, "context": str}
    """
    extraction_prompt = """Extract named entities from this text. Return ONLY valid JSON array.

Entity types: PERSON, ORG, DATE, LOCATION, PRODUCT, EVENT

Text: {text}

Return format (no other text):
[{{"name": "entity name", "type": "TYPE", "context": "brief relationship context"}}]"""

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": extraction_prompt.format(text=text[:1000])}],
            temperature=0,
            max_tokens=500
        )
        result = resp.choices[0].message.content.strip()
        # Find JSON array in response
        start = result.find('[')
        end = result.rfind(']') + 1
        if start >= 0 and end > start:
            return json.loads(result[start:end])
    except:
        pass
    return []


@dataclass
class ExtractedRelationship:
    """A relationship between entities extracted from text."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0


def extract_relationships_with_llm(text: str, client, model_name: str) -> Tuple[List[Dict], List[ExtractedRelationship]]:
    """
    Use LLM to extract entities AND their relationships.

    Returns (entities, relationships) where relationships are (subject, predicate, object) triples.
    """
    extraction_prompt = """Extract entities and relationships from this text. Return ONLY valid JSON.

Text: {text}

Return format:
{{
  "entities": [{{"name": "...", "type": "PERSON|ORG|DATE|LOCATION|PRODUCT"}}],
  "relationships": [{{"subject": "...", "predicate": "worked_at|founded|created|located_in|occurred_in|...", "object": "..."}}]
}}"""

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": extraction_prompt.format(text=text[:1500])}],
            temperature=0,
            max_tokens=800
        )
        result = resp.choices[0].message.content.strip()
        # Find JSON object in response
        start = result.find('{')
        end = result.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(result[start:end])
            entities = data.get("entities", [])
            relationships = [
                ExtractedRelationship(r["subject"], r["predicate"], r["object"])
                for r in data.get("relationships", [])
                if r.get("subject") and r.get("object")
            ]
            return entities, relationships
    except:
        pass
    return [], []


def extract_entities_smart(text: str, llm_entities: Optional[List[Dict]] = None) -> List[str]:
    """
    Extract entities, preferring LLM-extracted typed entities.
    Falls back to regex if no LLM entities provided.
    """
    if llm_entities:
        # Use LLM-extracted entities - already typed and filtered
        return [e["name"] for e in llm_entities if e.get("name")]

    # Fallback to regex
    return extract_entities(text)


def extract_with_relationships(text: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Extract entities AND sentence-level relationships.

    Returns (entities, relationships) where relationships are entity pairs
    that appeared in the same sentence (strong correlation signal).
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    all_entities = []
    sentence_pairs = []  # (entity1, entity2) pairs from same sentence

    for sentence in sentences:
        # Extract entities from this sentence
        sentence_entities = extract_entities(sentence)

        # All entities in same sentence are related
        for i, e1 in enumerate(sentence_entities):
            all_entities.append(e1)
            for e2 in sentence_entities[i+1:]:
                # Store as sorted pair for consistency
                pair = tuple(sorted([e1, e2]))
                sentence_pairs.append(pair)

    return list(set(all_entities)), sentence_pairs


# Relationship predicates we look for
RELATIONSHIP_PATTERNS = {
    'worked_at': [r'worked (at|for|with)', r'employed (at|by)', r'joined', r'was at', r'position at'],
    'founded': [r'founded', r'started', r'created', r'co-?founded', r'established'],
    'located_in': [r'based in', r'located in', r'headquartered in', r'from'],
    'occurred_in': [r'in (\d{4})', r'during', r'around'],
    'led': [r'led', r'headed', r'managed', r'directed', r'ran'],
    'developed': [r'developed', r'built', r'created', r'designed', r'architected'],
    'associated_with': [r'associated with', r'connected to', r'related to', r'involved with'],
}


def infer_relationship(sentence: str, entity1: str, entity2: str) -> Optional[str]:
    """
    Try to infer the relationship between two entities from sentence context.

    Returns predicate string like 'worked_at' or None if no clear relationship.
    """
    sentence_lower = sentence.lower()

    for predicate, patterns in RELATIONSHIP_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sentence_lower):
                return predicate

    return None


def extract_with_typed_relationships(text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Extract entities with TYPED relationships (subject, predicate, object).

    Returns (entities, relationships) where relationships are
    (entity1, predicate, entity2) tuples.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)

    all_entities = []
    typed_relationships = []  # (subject, predicate, object)

    for sentence in sentences:
        sentence_entities = extract_entities(sentence)

        for e in sentence_entities:
            all_entities.append(e)

        # Try to infer relationships between entity pairs
        for i, e1 in enumerate(sentence_entities):
            for e2 in sentence_entities[i+1:]:
                predicate = infer_relationship(sentence, e1, e2)
                if predicate:
                    # Order by type: person/org first, date last
                    if is_year(e2) and not is_year(e1):
                        typed_relationships.append((e1, predicate, e2))
                    elif is_year(e1) and not is_year(e2):
                        typed_relationships.append((e2, predicate, e1))
                    else:
                        typed_relationships.append((e1, predicate, e2))

    return list(set(all_entities)), typed_relationships


# Words to skip (generic, common, noise)
# Comprehensive list to filter sentence starters, pronouns, determiners, etc.
SKIP_WORDS = {
    # Sentence starters / transitions
    'based', 'here', 'however', 'therefore', 'while', 'after', 'before',
    'during', 'although', 'though', 'since', 'because', 'unless', 'until',
    'once', 'whereas', 'whereby', 'wherein', 'whereupon', 'meanwhile',
    'nonetheless', 'nevertheless', 'otherwise', 'furthermore', 'moreover',
    'additionally', 'similarly', 'consequently', 'subsequently', 'accordingly',
    'hence', 'thus', 'thereby', 'indeed', 'certainly', 'perhaps', 'possibly',
    'probably', 'actually', 'apparently', 'evidently', 'obviously', 'clearly',
    'specifically', 'particularly', 'especially', 'notably', 'importantly',
    'interestingly', 'surprisingly', 'unfortunately', 'fortunately', 'finally',
    'ultimately', 'essentially', 'basically', 'generally', 'typically',
    'alternatively', 'conversely', 'instead', 'rather', 'overall',

    # LLM-capitalized generic nouns (these get falsely capitalized in responses)
    'career', 'startups', 'startup', 'compensation', 'salary', 'benefits',
    'growth', 'opportunity', 'opportunities', 'risk', 'risks', 'challenge',
    'challenges', 'innovation', 'culture', 'cultural', 'moving', 'leaving',
    'working', 'personal', 'professional', 'development', 'experience',
    'expertise', 'skills', 'knowledge', 'learning', 'autonomy', 'flexibility',
    'uncertainty', 'stability', 'security', 'job', 'jobs', 'role', 'roles',
    'position', 'positions', 'team', 'teams', 'company', 'companies',
    'industry', 'industries', 'market', 'markets', 'technology', 'technologies',
    'software', 'hardware', 'engineering', 'product', 'products', 'service',
    'services', 'business', 'businesses', 'management', 'leadership',
    'networking', 'connections', 'relationships', 'impact', 'success',
    'failure', 'loss', 'gain', 'increased', 'decreased', 'improved',

    # Pronouns
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    'she', 'her', 'hers', 'herself', 'him', 'his', 'himself',
    'its', 'itself', 'one', 'ones', 'oneself',
    'who', 'whom', 'whose', 'which', 'that', 'what',
    'this', 'these', 'those', 'such', 'same',
    'anybody', 'anyone', 'anything', 'everybody', 'everyone', 'everything',
    'nobody', 'nothing', 'somebody', 'someone', 'something',
    'each', 'either', 'neither', 'both', 'few', 'many', 'most', 'several',
    'all', 'any', 'none', 'some', 'other', 'others', 'another',

    # Determiners / Articles
    'the', 'a', 'an', 'every', 'no', 'any', 'some', 'each',

    # Common verbs / auxiliaries
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'done',
    'be', 'been', 'being', 'was', 'were', 'are', 'is', 'am',
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    'get', 'got', 'getting', 'make', 'made', 'making',
    'let', 'say', 'said', 'says', 'saying', 'think', 'know', 'see', 'seem',

    # Prepositions / Conjunctions
    'with', 'without', 'from', 'for', 'and', 'but', 'not', 'nor', 'or',
    'about', 'above', 'across', 'against', 'along', 'among', 'around',
    'at', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'by', 'down', 'into', 'near', 'off', 'onto', 'out', 'over',
    'through', 'toward', 'under', 'upon', 'within',

    # Question words
    'what', 'when', 'where', 'why', 'how', 'whether',

    # Quantifiers / Numbers
    'one', 'two', 'three', 'four', 'five', 'first', 'second', 'third',
    'many', 'much', 'more', 'most', 'less', 'least', 'fewer',
    'several', 'various', 'numerous', 'multiple', 'single', 'double',

    # Generic nouns that aren't entities
    'people', 'person', 'thing', 'things', 'way', 'ways', 'time', 'times',
    'year', 'years', 'day', 'days', 'place', 'part', 'case', 'point',
    'fact', 'example', 'result', 'reason', 'question', 'answer', 'problem',
    'issue', 'area', 'number', 'group', 'level', 'type', 'kind', 'sort',
    'work', 'system', 'process', 'method', 'approach', 'practice',
    'information', 'data', 'evidence', 'source', 'report', 'study',

    # Adjectives that aren't meaningful
    'good', 'bad', 'new', 'old', 'great', 'small', 'large', 'big', 'little',
    'high', 'low', 'long', 'short', 'different', 'similar', 'same', 'own',
    'important', 'significant', 'major', 'minor', 'main', 'key', 'primary',
    'public', 'private', 'common', 'general', 'specific', 'particular',
    'certain', 'sure', 'likely', 'able', 'unable', 'possible', 'impossible',
    'available', 'necessary', 'required', 'relevant', 'related',

    # Adverbs
    'really', 'quite', 'rather', 'very', 'too', 'enough', 'almost', 'nearly',
    'just', 'only', 'even', 'still', 'yet', 'already', 'always', 'never',
    'often', 'sometimes', 'usually', 'rarely', 'seldom', 'ever',
    'now', 'then', 'soon', 'later', 'early', 'recently', 'currently',
    'please', 'also', 'well', 'back', 'away', 'together', 'apart',

    # Filler / hedge words
    'like', 'simply', 'merely', 'basically', 'essentially', 'primarily',
    'mainly', 'mostly', 'largely', 'partly', 'somewhat', 'fairly',

    # LLM response artifacts
    'note', 'recall', 'remember', 'imagine', 'consider', 'assume',
    'according', 'given', 'regarding', 'concerning', 'respect',
    'context', 'terms', 'sense', 'regard', 'relation', 'reference',
    'lack', 'tier', 'step', 'steps', 'item', 'items', 'code', 'better',

    # Time/frequency words that LLMs capitalize
    'weekly', 'daily', 'monthly', 'yearly', 'annual', 'annually',
    'quarterly', 'biweekly', 'hourly', 'regular', 'regularly',
    'ongoing', 'continuous', 'periodic', 'occasional', 'frequent',

    # Meeting/activity garbage
    'meeting', 'meetings', 'update', 'updates', 'discussion', 'discussions',
    'review', 'reviews', 'session', 'sessions', 'call', 'calls',
    'sync', 'syncs', 'standup', 'standups', 'retrospective', 'retro',
    'planning', 'grooming', 'sprint', 'sprints', 'iteration', 'iterations',

    # Document/communication garbage
    'email', 'emails', 'message', 'messages', 'document', 'documents',
    'memo', 'memos', 'report', 'reports', 'presentation', 'presentations',
    'slide', 'slides', 'agenda', 'agendas', 'summary', 'summaries',
    'notes', 'minutes', 'action', 'actions', 'followup', 'follow',

    # Generic corporate speak
    'stakeholder', 'stakeholders', 'deliverable', 'deliverables',
    'milestone', 'milestones', 'timeline', 'timelines', 'deadline', 'deadlines',
    'scope', 'requirement', 'requirements', 'specification', 'specifications',
    'priority', 'priorities', 'objective', 'objectives', 'goal', 'goals',
    'metric', 'metrics', 'kpi', 'kpis', 'target', 'targets',
    'resource', 'resources', 'budget', 'budgets', 'cost', 'costs',
    'effort', 'efforts', 'task', 'tasks', 'activity', 'activities',

    # Generic technical garbage
    'feature', 'features', 'bug', 'bugs', 'fix', 'fixes', 'patch', 'patches',
    'release', 'releases', 'version', 'versions', 'build', 'builds',
    'deploy', 'deployment', 'deployments', 'test', 'tests', 'testing',
    'integration', 'integrations', 'implementation', 'implementations',
    'architecture', 'design', 'designs', 'solution', 'solutions',
    'framework', 'frameworks', 'library', 'libraries', 'tool', 'tools',
    'api', 'apis', 'endpoint', 'endpoints', 'interface', 'interfaces',
    'database', 'databases', 'server', 'servers', 'client', 'clients',

    # More LLM-isms
    'individual', 'individuals', 'entity', 'entities', 'element', 'elements',
    'component', 'components', 'module', 'modules', 'unit', 'units',
    'instance', 'instances', 'object', 'objects', 'class', 'classes',
    'function', 'functions', 'variable', 'variables', 'parameter', 'parameters',
    'input', 'inputs', 'output', 'outputs', 'response', 'responses',
    'request', 'requests', 'query', 'queries', 'operation', 'operations',
}


@dataclass
class Concept:
    """A concept extracted from text with scoring metadata."""
    text: str
    word_count: int
    is_proper_noun: bool
    is_year: bool
    is_organization: bool

    @property
    def specificity(self) -> float:
        """Higher = more specific/complex concept."""
        score = 1.0

        # Multi-word concepts are more specific
        score *= (1 + (self.word_count - 1) * 0.5)

        # Proper nouns are more specific
        if self.is_proper_noun:
            score *= 1.3

        # Organizations are highly specific
        if self.is_organization:
            score *= 1.5

        # Years are specific data points
        if self.is_year:
            score *= 1.2

        return score


def is_year(text: str) -> bool:
    """Check if text is a year or decade."""
    return bool(re.match(r'^(19|20)\d{2}s?$', text))


def is_organization(text: str) -> bool:
    """Check if text looks like an organization name."""
    org_patterns = [
        r'\b(Inc|Corp|LLC|Ltd|Company|Software|Technologies|Systems|Labs?)\b',
        r'\b(University|Institute|Foundation|Association)\b',
        r'\b(Google|Microsoft|Apple|Amazon|Facebook|Meta)\b',
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in org_patterns)


def extract_concepts(text: str) -> List[Concept]:
    """
    Extract concepts from text, preferring complex multi-word concepts.

    Returns list of Concept objects with scoring metadata.
    """
    concepts = []

    # Pattern for multi-word proper noun phrases (capitalized sequences)
    # Excludes phrases starting with articles (The, A, An)
    multi_word_pattern = r'\b(?!(?:The|A|An)\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'

    # Pattern for single proper nouns
    single_proper_pattern = r'\b([A-Z][a-z]{2,})\b'

    # Pattern for years
    year_pattern = r'\b((?:19|20)\d{2}s?)\b'

    # Pattern for technical/domain phrases (lowercase multi-word)
    # Matches: "machine learning", "software engineer", "data pipeline"
    tech_phrase_pattern = r'\b((?:software|data|machine|deep|cloud|web|mobile|backend|frontend|full[- ]?stack|devops|site reliability|product|project|program|engineering|technical|senior|junior|lead|staff|principal|chief|head of)\s+(?:engineer(?:ing)?|developer|architect|manager|scientist|analyst|designer|lead|director|officer|learning|infrastructure|pipeline|platform|systems?|services?|operations?))\b'

    # Pattern for role titles
    role_pattern = r'\b((?:CEO|CTO|CFO|COO|VP|SVP|EVP|Director|Manager|Lead|Head|Chief|Principal|Senior|Staff)\s+(?:of\s+)?[A-Z]?[a-z]+(?:\s+[A-Z]?[a-z]+)*)\b'

    # Pattern for project/product names with mixed case
    # Matches: "iOS app", "API design", "ML model"
    mixed_pattern = r'\b([A-Z]{2,}(?:\s+[a-z]+)+|[a-z]+(?:\s+[A-Z]{2,}))\b'

    # Extract multi-word concepts first (higher priority)
    found_spans = set()

    for match in re.finditer(multi_word_pattern, text):
        concept_text = match.group(1)
        words = concept_text.split()

        # Skip if all words are in skip list
        if all(w.lower() in SKIP_WORDS for w in words):
            continue

        # Skip if too generic
        if len(words) == 2 and words[-1].lower() in {'the', 'a', 'an', 'of', 'in', 'on', 'at'}:
            continue

        found_spans.add((match.start(), match.end()))
        concepts.append(Concept(
            text=concept_text,
            word_count=len(words),
            is_proper_noun=True,
            is_year=False,
            is_organization=is_organization(concept_text),
        ))

    # Extract technical/domain phrases
    for match in re.finditer(tech_phrase_pattern, text, re.IGNORECASE):
        concept_text = match.group(1).lower()
        # Skip if overlaps with existing
        if any(match.start() >= s and match.end() <= e for s, e in found_spans):
            continue

        found_spans.add((match.start(), match.end()))
        concepts.append(Concept(
            text=concept_text,
            word_count=len(concept_text.split()),
            is_proper_noun=False,
            is_year=False,
            is_organization=False,
        ))

    # Extract role titles
    for match in re.finditer(role_pattern, text):
        concept_text = match.group(1)
        if any(match.start() >= s and match.end() <= e for s, e in found_spans):
            continue

        found_spans.add((match.start(), match.end()))
        concepts.append(Concept(
            text=concept_text,
            word_count=len(concept_text.split()),
            is_proper_noun=True,
            is_year=False,
            is_organization=False,
        ))

    # Extract single proper nouns (not already part of multi-word)
    for match in re.finditer(single_proper_pattern, text):
        # Skip if this span overlaps with a multi-word concept
        if any(match.start() >= s and match.end() <= e for s, e in found_spans):
            continue

        word = match.group(1)

        # Skip common words
        if word.lower() in SKIP_WORDS:
            continue

        # Skip very short words
        if len(word) < 3:
            continue

        concepts.append(Concept(
            text=word,
            word_count=1,
            is_proper_noun=True,
            is_year=False,
            is_organization=is_organization(word),
        ))

    # Extract years
    for match in re.finditer(year_pattern, text):
        year_text = match.group(1)
        concepts.append(Concept(
            text=year_text,
            word_count=1,
            is_proper_noun=False,
            is_year=True,
            is_organization=False,
        ))

    return concepts


def extract_entities(text: str) -> List[str]:
    """
    Simple entity extraction - returns list of entity strings.

    For backward compatibility with existing code.
    Prefers complex concepts over simple words.
    """
    concepts = extract_concepts(text)

    # Sort by specificity (complex concepts first)
    concepts.sort(key=lambda c: c.specificity, reverse=True)

    # Return unique entity texts
    seen = set()
    entities = []
    for c in concepts:
        if c.text not in seen:
            seen.add(c.text)
            entities.append(c.text)

    return entities


def score_concept(concept: str, frequency: int, connections: int) -> float:
    """
    Score a concept based on:
    - frequency: how often it appears
    - specificity: complexity/uniqueness (multi-word, proper noun, etc.)
    - connectivity: how many other concepts it co-occurs with

    Higher score = more important concept for narrative.
    """
    # Estimate specificity from text
    words = concept.split()
    word_count = len(words)

    specificity = 1.0 + (word_count - 1) * 0.5
    if is_organization(concept):
        specificity *= 1.5
    if is_year(concept):
        specificity *= 1.2

    # Combined score
    return frequency * specificity * (1 + connections * 0.3)
