"""
Document Parser - Parse documents, detect gaps/redactions.

LLM handles understanding content. We just find redaction markers.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Gap:
    """A detected gap/redaction in a document."""
    id: int
    start: int
    end: int
    original_text: str  # The redaction marker itself ([REDACTED], etc.)
    context_before: str
    context_after: str
    gap_type: str  # redaction, placeholder, ellipsis


@dataclass
class ParsedDocument:
    """A parsed document."""
    content: str
    content_type: str  # text, pdf, image
    gaps: list[Gap] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# Redaction detection patterns - these are actual markers, not content extraction
REDACTION_PATTERNS = [
    (r'\[REDACTED\]', 'redaction'),
    (r'\[REDACTED[^\]]*\]', 'redaction'),
    (r'\(REDACTED\)', 'redaction'),
    (r'<REDACTED>', 'redaction'),
    (r'XXXXX+', 'redaction'),
    (r'█+', 'redaction'),
    (r'_{5,}', 'placeholder'),
    (r'\.{5,}', 'placeholder'),
    (r'\[\s*\?\s*\]', 'placeholder'),
    (r'\[blank\]', 'placeholder', re.IGNORECASE),
    (r'\[name\]', 'placeholder', re.IGNORECASE),
    (r'\[NAME REDACTED\]', 'redaction', re.IGNORECASE),
    (r'\[\*+\]', 'redaction'),
    (r'<<[^>]*>>', 'placeholder'),
]


def parse_document(content: str, content_type: str = 'text') -> ParsedDocument:
    """Parse a document - just find gaps, LLM understands the rest."""
    doc = ParsedDocument(content=content, content_type=content_type)
    doc.gaps = detect_gaps(content)
    return doc


def detect_gaps(content: str, context_window: int = 300) -> list[Gap]:
    """Detect gaps/redactions in document content."""
    gaps = []
    gap_id = 0

    for pattern_info in REDACTION_PATTERNS:
        if len(pattern_info) == 2:
            pattern, gap_type = pattern_info
            flags = 0
        else:
            pattern, gap_type, flags = pattern_info

        for match in re.finditer(pattern, content, flags):
            start = match.start()
            end = match.end()

            context_before = content[max(0, start - context_window):start].strip()
            context_after = content[end:end + context_window].strip()

            gap = Gap(
                id=gap_id,
                start=start,
                end=end,
                original_text=match.group(),
                context_before=context_before,
                context_after=context_after,
                gap_type=gap_type,
            )
            gaps.append(gap)
            gap_id += 1

    gaps.sort(key=lambda g: g.start)
    return gaps


def attempt_pdf_bypass(pdf_path: str) -> dict:
    """
    Attempt to extract text hidden under redactions.

    Many PDFs have 'fake' redactions - black boxes overlaid on text.
    The text layer is still accessible.
    """
    results = {
        'bypasses_found': [],
        'metadata': {},
        'hidden_text': []
    }

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[BYPASS] PyMuPDF not installed, skipping PDF bypass")
        return results

    try:
        doc = fitz.open(pdf_path)

        # Extract ALL text including under redactions
        for page_num, page in enumerate(doc):
            # Get text blocks
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                # Check if this text is under a redaction annotation
                                rect = fitz.Rect(span["bbox"])
                                annots = page.annots()
                                if annots:
                                    for annot in annots:
                                        if annot.type[0] == 12:  # Redaction annotation
                                            if rect.intersects(annot.rect):
                                                results['bypasses_found'].append({
                                                    'page': page_num + 1,
                                                    'text': text,
                                                    'method': 'text_layer_extraction'
                                                })

        # Extract metadata
        results['metadata'] = dict(doc.metadata) if doc.metadata else {}

        doc.close()

    except Exception as e:
        print(f"[BYPASS] PDF extraction error: {e}")

    return results


def attempt_text_bypass(content: str, gap: Gap) -> Optional[str]:
    """
    Check if the 'redacted' marker actually contains hidden text.

    Some copy-pasted content from PDFs includes the hidden text.
    """
    # Check if the gap marker itself contains readable text
    marker = gap.original_text

    # Sometimes redactions are just Unicode boxes but text is adjacent
    # or embedded in weird ways

    # Check for zero-width characters or hidden text
    import unicodedata

    hidden_chars = []
    for char in marker:
        if unicodedata.category(char) in ('Cf', 'Zs', 'Cc'):
            # Control format, space separator, or control char
            continue
        if char not in '█▓▒░_.[]*<>()REDACTED ':
            hidden_chars.append(char)

    if hidden_chars and len(hidden_chars) > 2:
        return ''.join(hidden_chars)

    return None


def extract_context(doc: ParsedDocument, gap: Gap, window: int = 500) -> dict:
    """Extract context around a gap."""
    start = gap.start
    end = gap.end

    wide_before = doc.content[max(0, start - window):start]
    wide_after = doc.content[end:end + window]

    return {
        'context_before': gap.context_before,
        'context_after': gap.context_after,
        'wide_context_before': wide_before,
        'wide_context_after': wide_after,
        'full_context': f"{gap.context_before} [???] {gap.context_after}",
    }
