"""
Document Cleaner - Use LLM to clean up garbage OCR and evaluate research quality.
"""

from typing import Optional


def is_useful_research(content: str, title: str, topic: str) -> tuple[bool, str]:
    """
    Use LLM to evaluate if research content is useful for the topic.
    Returns (is_useful, reason).
    """
    if not content or len(content) < 100:
        return False, "Too short"

    # Quick heuristics first
    lower = content.lower()
    trash_signals = [
        "just a moment", "checking your browser", "access denied",
        "403 forbidden", "404 not found", "subscribe to read",
        "cookies must be enabled", "please enable javascript",
        "captcha", "cloudflare"
    ]
    for signal in trash_signals:
        if signal in lower[:500]:
            return False, f"Blocked/paywall: {signal}"

    try:
        from config import get_client
        client, config = get_client('groq/llama-3.1-8b-instant')

        # Quick relevance check - BE LENIENT, any info about the topic is potentially useful
        response = client.chat.completions.create(
            model=config['model'],
            messages=[{
                "role": "user",
                "content": f"""Is this content useful for researching: "{topic}"?

TITLE: {title}
CONTENT (first 1500 chars):
{content[:1500]}

BE LENIENT - even small mentions, profile snippets, forum posts, or partial info can be valuable for OSINT.
Only mark as TRASH if it's truly irrelevant (wrong person entirely) or blocked content.

Reply with ONLY one of:
- USEFUL: [brief reason - even partial info counts]
- TRASH: [only if completely irrelevant or blocked]"""
            }],
            max_tokens=100,
            temperature=0.1
        )

        result = response.choices[0].message.content.strip()
        if result.startswith("USEFUL"):
            return True, result
        else:
            return False, result

    except Exception as e:
        # On error, keep the content (don't filter)
        return True, f"Eval failed: {e}"


def suggest_next_searches(topic: str, entities: list[str], existing_research: str) -> list[str]:
    """
    Use LLM to suggest what to search next based on what we've found.
    Returns list of search queries.
    """
    try:
        from config import get_client
        client, config = get_client('groq/llama-3.1-8b-instant')

        response = client.chat.completions.create(
            model=config['model'],
            messages=[{
                "role": "user",
                "content": f"""Topic: {topic}

Known entities: {', '.join(entities[:15])}

Existing research summary (last 500 chars):
{existing_research[-500:] if existing_research else "None yet"}

Suggest 5 specific web searches to find MORE information about this topic.

Adapt your searches to the topic type:
- For usernames/handles: use quotes and site: operators (e.g. "handle" site:twitter.com)
- For people: name + location, employer, news, court records
- For companies: lawsuits, investigations, SEC filings, news
- For places/events: news coverage, official records, investigations

Be specific. Use quotes for exact phrases. Include names and dates where possible.

Reply with ONLY the search queries, one per line."""
            }],
            max_tokens=300,
            temperature=0.7
        )

        queries = [
            line.strip().lstrip('0123456789.-) ')
            for line in response.choices[0].message.content.strip().split('\n')
            if line.strip() and len(line.strip()) > 10
        ]
        return queries[:5]

    except Exception as e:
        print(f"[RESEARCH] suggest_next_searches failed: {e}")
        return []


def looks_like_garbage(text: str) -> bool:
    """Check if text has OCR garbage."""
    if not text:
        return False
    garbage_chars = text.count('ï¿½') + text.count('�') + text.count('\ufffd')
    weird_ratio = garbage_chars / max(len(text), 1)
    return weird_ratio > 0.01 or garbage_chars > 50


def clean_with_llm(content: str, title: str) -> Optional[str]:
    """
    Use LLM to clean up garbage OCR text.
    Returns None if document is unreadable (should be discarded).
    """
    if not content:
        return content

    if not looks_like_garbage(content):
        return content  # Already clean

    try:
        from config import get_client
        client, config = get_client('groq/llama-3.3-70b-versatile')

        chunk = content[:8000]

        response = client.chat.completions.create(
            model=config['model'],
            messages=[{
                "role": "user",
                "content": f"""This is OCR text from "{title}". Extract ONLY the readable parts.

RULES:
- Output ONLY text you can actually read
- Remove all garbage characters (ï¿½, �, random symbols)
- If a section is unreadable, skip it entirely
- If the whole thing is unreadable, just say "[UNREADABLE]"
- Format readable parts as clean markdown
- Preserve names, dates, locations, any real content

TEXT:
{chunk}

CLEANED OUTPUT:"""
            }],
            max_tokens=4000,
            temperature=0.1
        )

        cleaned = response.choices[0].message.content.strip()

        # Check if cleanup worked
        if looks_like_garbage(cleaned) or cleaned == "[UNREADABLE]":
            print(f"[RESEARCH] Discarding unreadable: {title[:40]}...")
            return None

        print(f"[RESEARCH] Cleaned OCR: {title[:40]}...")

        if len(content) > 8000:
            cleaned += f"\n\n---\n*[Truncated - original {len(content)} chars]*"

        return cleaned

    except Exception as e:
        print(f"[RESEARCH] LLM cleanup failed: {e}")
        return content
