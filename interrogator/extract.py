"""
Entity and concept extraction from text responses.

Prefers complex, specific concepts over generic words:
- "Fog Creek Software" > "Software"
- "Extreme Programming" > "Programming"
- "Kansas City" > "Kansas"
"""

import re
from typing import List, Set, Tuple
from dataclasses import dataclass


# Words to skip (generic, common, noise)
SKIP_WORDS = {
    'based', 'here', 'however', 'therefore', 'while', 'after', 'before',
    'without', 'about', 'some', 'public', 'his', 'her', 'their', 'would',
    'could', 'should', 'please', 'keep', 'even', 'also', 'just', 'still',
    'yet', 'already', 'always', 'never', 'often', 'sometimes', 'usually',
    'the', 'this', 'that', 'with', 'from', 'for', 'and', 'but', 'not',
    'have', 'has', 'had', 'been', 'being', 'was', 'were', 'are', 'is',
    'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
    'can', 'may', 'might', 'must', 'shall', 'will', 'according', 'given',
    'one', 'two', 'three', 'many', 'much', 'more', 'most', 'other', 'another',
    'same', 'different', 'such', 'these', 'those', 'each', 'every', 'any',
    'all', 'both', 'few', 'several', 'own', 'able', 'unable', 'likely',
    'specifically', 'additionally', 'similarly', 'furthermore', 'moreover',
    'indeed', 'certainly', 'perhaps', 'possibly', 'probably', 'actually',
    'really', 'quite', 'rather', 'very', 'too', 'enough', 'almost', 'nearly',
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
