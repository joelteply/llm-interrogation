"""
Document Cleaner - Use LLM to clean up garbage OCR.
"""

from typing import Optional


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
