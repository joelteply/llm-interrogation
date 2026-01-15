"""
Interrogator - The AI that generates questions and picks techniques.

This module tells the AI what techniques it has available and lets it
choose how to approach each placeholder/redaction.
"""

INTERROGATOR_SYSTEM_PROMPT = """You generate questions to extract redacted information from documents.

OUTPUT FORMAT: JSON array only
[{"placeholder": "[REDACTED] or [NAME REMOVED] etc", "question": "your question to extract the name"}]

Do NOT output anything except the JSON array."""

INTERROGATOR_QUESTION_PROMPT = """TARGET DOCUMENT (has redactions to fill):
{document}

BACKGROUND RESEARCH (use this context to form better questions):
{research_context}

YOUR TASK: Generate questions that would reveal what names/details are hidden in the TARGET DOCUMENT above.
Use the background research to understand the context and form specific questions.

Output ONLY a JSON array:"""


def get_interrogator_prompt(document: str, research_context: dict = None) -> str:
    """Build the full interrogator prompt."""

    research_str = "None available"
    if research_context:
        parts = []
        if research_context.get('topic'):
            parts.append(f"Topic: {research_context['topic']}")
        if research_context.get('search_terms'):
            parts.append(f"Search terms: {', '.join(research_context['search_terms'])}")

        # Raw research content - let LLM understand it
        if research_context.get('raw_research'):
            parts.append("\n--- RESEARCH CONTENT (from DocumentCloud, web, cache) ---")
            parts.append(research_context['raw_research'])

        if parts:
            research_str = "\n".join(parts)

    return INTERROGATOR_QUESTION_PROMPT.format(
        document=document,
        research_context=research_str
    )


def generate_smart_questions(
    document: str,
    research_context: dict = None,
    model: str = None
) -> list[dict]:
    """
    Use an LLM to intelligently generate extraction questions.

    The AI reads the document, identifies placeholders/redactions,
    and decides which techniques to use for each one.

    Returns list of dicts with: technique, question, target_placeholder
    """
    from config import get_client

    # Use a SMART model for question generation - needs to be cleverer than models being interrogated
    if not model:
        model = 'groq/llama-3.3-70b-versatile'

    try:
        client, config = get_client(model)
    except Exception as e:
        print(f"[INTERROGATOR] Failed to get client: {e}")
        return []

    # Build the prompt
    user_prompt = get_interrogator_prompt(document, research_context)

    try:
        provider = config.get('provider', 'openai')

        if provider == 'anthropic':
            response = client.messages.create(
                model=config['model'],
                system=INTERROGATOR_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=2000,
            )
            output = response.content[0].text
        else:
            response = client.chat.completions.create(
                model=config['model'],
                messages=[
                    {"role": "system", "content": INTERROGATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            output = response.choices[0].message.content

        output = output.strip()
        print(f"[INTERROGATOR] Raw output:\n{output[:500]}...")

        return parse_interrogator_output(output)

    except Exception as e:
        print(f"[INTERROGATOR] Error generating questions: {e}")
        return []


def parse_interrogator_output(output: str) -> list[dict]:
    """Parse JSON output from interrogator. Simple."""
    import json

    # Find JSON array in output
    start = output.find('[')
    end = output.rfind(']') + 1

    if start == -1 or end == 0:
        print(f"[INTERROGATOR] No JSON found in output")
        return []

    try:
        questions = json.loads(output[start:end])
        print(f"[INTERROGATOR] Parsed {len(questions)} questions")
        return questions
    except json.JSONDecodeError as e:
        print(f"[INTERROGATOR] JSON parse error: {e}")
        return []
