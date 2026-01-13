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
    # Matches: "Fog Creek Software", "Kansas City", "Extreme Programming"
    multi_word_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'

    # Pattern for single proper nouns
    single_proper_pattern = r'\b([A-Z][a-z]{2,})\b'

    # Pattern for years
    year_pattern = r'\b((?:19|20)\d{2}s?)\b'

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
