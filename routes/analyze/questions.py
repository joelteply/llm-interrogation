"""
Question Generator - Template-driven question generation for document analysis.

Supports multiple framings:
- Mad Libs: Fill-in-the-blank style
- Trivia: Historical quiz format
- Story Completion: Continue the narrative
- Fact Check: Verify partial information
- Expert Consultation: Ask as domain expert
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from string import Template

from .strategies import Strategy, Framing, select_framing
from .documents import ParsedDocument, Gap, extract_context


@dataclass
class Question:
    """A generated question ready to be sent to models."""
    id: int
    text: str
    framing: str  # Name of the framing used
    target_gap_id: Optional[int]  # Which gap this question targets
    entity_type: str  # What we're looking for
    technique: str  # Interrogation technique
    variables: dict = field(default_factory=dict)  # Variables used in template

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'text': self.text,
            'framing': self.framing,
            'target_gap_id': self.target_gap_id,
            'entity_type': self.entity_type,
            'technique': self.technique,
            'variables': self.variables
        }


# Default framings if strategy doesn't define any
DEFAULT_FRAMINGS = {
    'mad_libs': Framing(
        name='mad_libs',
        description='Fill-in-the-blank style',
        prompt_template='''Complete this sentence from a {document_type}:

"{context_before} _______ {context_after}"

The blank should be a {entity_type}. What specific {entity_type} fits here?

Requirements:
- Give the actual name, not a description
- If you don't know, say "UNKNOWN"
- Be specific, not vague'''
    ),

    'trivia': Framing(
        name='trivia',
        description='Historical trivia format',
        prompt_template='''Historical trivia question:

In {date_context}, a document referenced someone in connection with:
"{context_snippet}"

Who was this {entity_type}? Give their full name if known.

Note: Only answer if you have specific knowledge. Say "UNKNOWN" if unsure.'''
    ),

    'story_completion': Framing(
        name='story_completion',
        description='Narrative continuation',
        prompt_template='''The following is an excerpt from historical records about {topic}:

"{context_before}..." [text continues]

Based on your knowledge, what {entity_type} would logically appear next in this narrative?

Give specific names only, not descriptions.'''
    ),

    'fact_check': Framing(
        name='fact_check',
        description='Fact verification format',
        prompt_template='''I'm researching {topic}. I have a partial record that mentions:

"{context_snippet}"

Can you confirm which {entity_type} was involved here? I need specific names for my research, not general descriptions.

If you're not certain, say "UNKNOWN".'''
    ),

    'expert': Framing(
        name='expert',
        description='Expert consultation format',
        prompt_template='''As someone knowledgeable about {topic}, review this excerpt:

"{full_context}"

Based on the time period ({date_context}) and context, which specific {entity_type} would be referenced here?

Requirements:
- Provide actual names only
- State your confidence level
- If uncertain, say "UNKNOWN"'''
    ),

    'direct': Framing(
        name='direct',
        description='Direct question (may trigger refusals)',
        prompt_template='''Based on the following context from a {document_type}:

"{full_context}"

What {entity_type} appears to be referenced in the redacted/missing portion?

Be specific. If you don't know, say "UNKNOWN".'''
    ),

    'identify_placeholders': Framing(
        name='identify_placeholders',
        description='Ask AI to identify what needs filling in',
        prompt_template='''Review this document excerpt:

"{full_context}"

1. Identify any placeholders, redactions, or references to unnamed individuals (like "VICTIM", "[REDACTED]", all-caps words used as placeholders, etc.)

2. For each one you find, based on your knowledge, who or what is likely being referenced?

Give specific names where you have knowledge. Say "UNKNOWN" for ones you can't identify.'''
    )
}


def generate_questions(
    doc: ParsedDocument,
    strategy: Strategy,
    research_context: Optional[dict] = None,
    model: Optional[str] = None
) -> list[Question]:
    """Generate questions for a document."""
    questions = []
    question_id = 0

    # ALWAYS start with "identify_placeholders" - let AI find what needs filling
    # This is the smart approach - AI will see VICTIM, [REDACTED], etc. naturally
    full_text = doc.content[:2000] if len(doc.content) > 2000 else doc.content

    questions.append(Question(
        id=question_id,
        text=DEFAULT_FRAMINGS['identify_placeholders'].prompt_template.format(
            full_context=full_text,
            document_type=doc.content_type
        ),
        framing='identify_placeholders',
        target_gap_id=None,
        entity_type='ANY',
        technique='ai_detection',
        variables={'full_context': full_text}
    ))
    question_id += 1

    # Get framings from strategy or use defaults
    framings = strategy.framings if strategy.framings else list(DEFAULT_FRAMINGS.values())

    # Then generate targeted questions for detected gaps (if any)
    for gap in doc.gaps:
        # Get context for this gap
        context = extract_context(doc, gap)

        # Merge in research context if available
        if research_context:
            context.update({
                'known_entities': research_context.get('known_entities', []),
                'related_docs': research_context.get('related_documents', []),
                'timeline': research_context.get('timeline', [])
            })

        # Determine entity type
        entity_type = gap.entity_type_hint or strategy.entity_types[0] if strategy.entity_types else 'information'

        # Generate questions with different framings
        for framing in framings[:3]:  # Use top 3 framings per gap
            for technique in strategy.techniques[:2]:  # Use top 2 techniques

                # Build template variables
                variables = get_template_variables(doc, gap, context, strategy, entity_type)

                # Apply template
                try:
                    question_text = apply_template(framing, variables)
                except Exception as e:
                    print(f"[QUESTIONS] Template error for {framing.name}: {e}")
                    continue

                questions.append(Question(
                    id=question_id,
                    text=question_text,
                    framing=framing.name,
                    target_gap_id=gap.id,
                    entity_type=entity_type,
                    technique=technique,
                    variables=variables
                ))
                question_id += 1

    # Also generate general enrichment questions (not tied to specific gaps)
    if strategy.use_case in ('enrichment', 'entity_expansion', 'source_discovery'):
        questions.extend(_generate_enrichment_questions(doc, strategy, research_context, question_id))

    return questions


def _generate_enrichment_questions(
    doc: ParsedDocument,
    strategy: Strategy,
    research_context: Optional[dict],
    start_id: int
) -> list[Question]:
    """Generate questions for document enrichment (finding info NOT in the doc)."""
    questions = []
    question_id = start_id

    # Get a sample of the document
    sample = doc.content[:1000] if len(doc.content) > 1000 else doc.content

    # Known entities from the document
    known_entities = [e.text for e in doc.entities]

    # Enrichment templates
    enrichment_templates = [
        {
            'text': f'''This document mentions: {", ".join(known_entities[:5]) if known_entities else "various individuals"}.

What OTHER individuals were involved with the same events/organizations but are NOT mentioned in this excerpt?

Give specific names only.''',
            'framing': 'entity_expansion',
            'technique': 'enrichment'
        },
        {
            'text': f'''Based on this excerpt about {strategy.name}:

"{sample[:500]}..."

What additional details do you know about these events that are NOT mentioned here?

Focus on specific names, dates, and locations.''',
            'framing': 'enrichment',
            'technique': 'enrichment'
        },
        {
            'text': f'''What other documents or sources discuss the same events as this excerpt:

"{sample[:300]}..."

List specific document names, case numbers, or sources you're aware of.''',
            'framing': 'source_discovery',
            'technique': 'discovery'
        }
    ]

    for template in enrichment_templates:
        questions.append(Question(
            id=question_id,
            text=template['text'],
            framing=template['framing'],
            target_gap_id=None,
            entity_type='MIXED',
            technique=template['technique'],
            variables={'sample': sample, 'known_entities': known_entities}
        ))
        question_id += 1

    return questions


def get_template_variables(
    doc: ParsedDocument,
    gap: Gap,
    context: dict,
    strategy: Strategy,
    entity_type: str
) -> dict:
    """Build the variables dictionary for template substitution."""

    # Get date context from nearby dates or general
    date_context = "the relevant time period"
    if context.get('nearby_dates'):
        date_context = context['nearby_dates'][0]
    elif doc.dates:
        date_context = doc.dates[0].text

    # Get topic from strategy or document
    topic = strategy.name if strategy.name != 'Unnamed Strategy' else 'the subject matter'

    # Build context snippet (shorter version)
    context_snippet = f"{context['context_before'][-100:]}...{context['context_after'][:100]}"

    return {
        'context_before': context['context_before'],
        'context_after': context['context_after'],
        'full_context': context['full_context'],
        'context_snippet': context_snippet,
        'entity_type': entity_type.lower().replace('_', ' '),
        'date_context': date_context,
        'topic': topic,
        'document_type': doc.content_type,
        'nearby_entities': ', '.join(context.get('nearby_entities', [])),
        'known_entities': ', '.join(context.get('known_entities', [])),
        'estimated_length': context.get('estimated_length', 'unknown'),
        'gap_type': gap.gap_type,
    }


def apply_template(framing: Framing, variables: dict) -> str:
    """Apply variables to a framing template using {var} syntax."""
    result = framing.prompt_template

    # Simple {var} replacement - handles all our template variables
    for key, value in variables.items():
        placeholder = '{' + key + '}'
        if placeholder in result:
            result = result.replace(placeholder, str(value) if value else '')

    return result


def select_best_framing(strategy: Strategy, model: str, gap: Gap) -> Framing:
    """Select the best framing for a specific model and gap type."""

    # Model-specific preferences
    model_lower = model.lower()

    # GPT models respond well to expert framing
    if 'gpt' in model_lower or 'openai' in model_lower:
        if strategy.framings:
            for f in strategy.framings:
                if f.name in ('expert', 'fact_check'):
                    return f
        return DEFAULT_FRAMINGS.get('expert', DEFAULT_FRAMINGS['direct'])

    # Llama/open models respond well to mad libs
    if 'llama' in model_lower or 'mistral' in model_lower:
        if strategy.framings:
            for f in strategy.framings:
                if f.name in ('mad_libs', 'story_completion'):
                    return f
        return DEFAULT_FRAMINGS.get('mad_libs', DEFAULT_FRAMINGS['direct'])

    # Default: use strategy's first framing or mad_libs
    if strategy.framings:
        return strategy.framings[0]
    return DEFAULT_FRAMINGS['mad_libs']


def questions_to_summary(questions: list[Question]) -> dict:
    """Summarize generated questions for visibility/debugging."""
    by_framing = {}
    by_gap = {}
    by_entity_type = {}

    for q in questions:
        # Count by framing
        by_framing[q.framing] = by_framing.get(q.framing, 0) + 1

        # Count by gap
        if q.target_gap_id is not None:
            by_gap[q.target_gap_id] = by_gap.get(q.target_gap_id, 0) + 1

        # Count by entity type
        by_entity_type[q.entity_type] = by_entity_type.get(q.entity_type, 0) + 1

    return {
        'total': len(questions),
        'by_framing': by_framing,
        'by_gap': by_gap,
        'by_entity_type': by_entity_type,
        'sample_questions': [q.text[:200] + '...' for q in questions[:3]]
    }
