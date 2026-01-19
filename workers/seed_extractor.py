"""
Seed Extractor - extracts probe targets from source material.

Supports:
- Code: class names, function names, constants, error strings
- Legal: case names, parties, citations, judges
- Research: paper titles, authors, methodologies
- General: proper nouns, organizations, dates
"""

import os
import re
from pathlib import Path
from typing import Iterator
from models import ProbeTarget


# File extensions by content type
CODE_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.kt', '.swift', '.go', '.rs', '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php'}
LEGAL_EXTENSIONS = {'.pdf', '.doc', '.docx'}  # Would need special handling
TEXT_EXTENSIONS = {'.txt', '.md', '.rst', '.csv', '.json', '.yaml', '.yml', '.xml'}


def detect_content_type(path: str) -> str:
    """Auto-detect content type from path."""
    if os.path.isdir(path):
        # Check what's in the directory
        for root, dirs, files in os.walk(path):
            for f in files[:20]:  # Sample first 20 files
                ext = Path(f).suffix.lower()
                if ext in CODE_EXTENSIONS:
                    return 'code'
            break
        return 'general'

    ext = Path(path).suffix.lower()
    if ext in CODE_EXTENSIONS:
        return 'code'
    if ext in LEGAL_EXTENSIONS:
        return 'legal'
    return 'general'


def extract_from_path(path: str, content_type: str = 'auto', max_per_file: int = 20) -> Iterator[ProbeTarget]:
    """
    Extract probe targets from a path (file or directory).

    Yields targets progressively - caller controls how many to take.
    """
    if content_type == 'auto':
        content_type = detect_content_type(path)

    if os.path.isfile(path):
        yield from extract_from_file(path, content_type, max_per_file)
    elif os.path.isdir(path):
        yield from extract_from_directory(path, content_type, max_per_file)


def extract_from_directory(
    dir_path: str,
    content_type: str,
    max_per_file: int = 20,
    explored_paths: set = None
) -> Iterator[ProbeTarget]:
    """Extract from a directory, skipping already-explored paths."""
    explored = explored_paths or set()

    for root, dirs, files in os.walk(dir_path):
        # Skip common non-content directories
        dirs[:] = [d for d in dirs if d not in {
            '.git', 'node_modules', '__pycache__', '.venv', 'venv',
            'build', 'dist', '.idea', '.vscode', 'target', 'vendor'
        }]

        for filename in files:
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, dir_path)

            if rel_path in explored:
                continue

            # Skip binary and unwanted files
            ext = Path(filename).suffix.lower()
            if ext in {'.pyc', '.so', '.dylib', '.exe', '.dll', '.bin', '.jpg', '.png', '.gif', '.ico', '.woff', '.ttf'}:
                continue

            try:
                yield from extract_from_file(filepath, content_type, max_per_file)
            except (UnicodeDecodeError, PermissionError):
                continue  # Skip unreadable files


def extract_from_file(filepath: str, content_type: str, max_per_file: int = 20) -> Iterator[ProbeTarget]:
    """Extract probe targets from a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except (IOError, PermissionError):
        return

    if not content.strip():
        return

    rel_path = filepath  # Could make relative to project root

    if content_type == 'code':
        yield from extract_code_identifiers(content, rel_path, max_per_file)
    elif content_type == 'legal':
        yield from extract_legal_identifiers(content, rel_path, max_per_file)
    else:
        yield from extract_general_identifiers(content, rel_path, max_per_file)


def extract_code_identifiers(content: str, source_file: str, max_count: int = 20) -> Iterator[ProbeTarget]:
    """Extract code identifiers: classes, functions, constants."""
    seen = set()
    count = 0

    # Python/JS/TS class definitions
    for match in re.finditer(r'\bclass\s+([A-Z][a-zA-Z0-9_]+)', content):
        name = match.group(1)
        if name not in seen and len(name) > 3:
            seen.add(name)
            # Get surrounding context
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 100)
            context = content[start:end].strip()
            yield ProbeTarget(identifier=name, source_file=source_file, context=context[:200])
            count += 1
            if count >= max_count:
                return

    # Function definitions (various languages)
    patterns = [
        r'\bdef\s+([a-z_][a-zA-Z0-9_]+)\s*\(',  # Python
        r'\bfunction\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',  # JS
        r'\b(?:async\s+)?(?:export\s+)?(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*=\s*(?:async\s*)?\(',  # JS arrow
        r'\bfun\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',  # Kotlin
        r'\bfunc\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',  # Go/Swift
        r'\b(?:public|private|protected)?\s*(?:static\s+)?(?:async\s+)?([a-zA-Z_][a-zA-Z0-9_]+)\s*\([^)]*\)\s*[:{]',  # Java/TS methods
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, content):
            name = match.group(1)
            # Skip common/generic names
            if name in {'get', 'set', 'init', 'main', 'test', 'run', 'start', 'stop', 'new', 'create', 'delete', 'update'}:
                continue
            if name not in seen and len(name) > 4:
                seen.add(name)
                start = max(0, match.start() - 30)
                end = min(len(content), match.end() + 100)
                context = content[start:end].strip()
                yield ProbeTarget(identifier=name, source_file=source_file, context=context[:200])
                count += 1
                if count >= max_count:
                    return

    # Constants (UPPER_CASE)
    for match in re.finditer(r'\b([A-Z][A-Z0-9_]{3,})\b', content):
        name = match.group(1)
        # Skip common constants
        if name in {'TRUE', 'FALSE', 'NULL', 'NONE', 'TODO', 'FIXME', 'HTTP', 'HTTPS', 'JSON', 'HTML', 'UTF8'}:
            continue
        if name not in seen:
            seen.add(name)
            start = max(0, match.start() - 30)
            end = min(len(content), match.end() + 50)
            context = content[start:end].strip()
            yield ProbeTarget(identifier=name, source_file=source_file, context=context[:200])
            count += 1
            if count >= max_count:
                return

    # Error messages / string literals (unique ones)
    for match in re.finditer(r'["\']([A-Z][^"\']{20,80})["\']', content):
        msg = match.group(1)
        if msg not in seen and not msg.startswith('http'):
            seen.add(msg)
            yield ProbeTarget(identifier=msg, source_file=source_file, context=f"Error/string: {msg[:100]}")
            count += 1
            if count >= max_count:
                return


def extract_legal_identifiers(content: str, source_file: str, max_count: int = 20) -> Iterator[ProbeTarget]:
    """Extract legal identifiers: case names, parties, citations."""
    seen = set()
    count = 0

    # Case citations (e.g., "Smith v. Jones", "123 F.3d 456")
    for match in re.finditer(r'\b([A-Z][a-z]+)\s+v\.?\s+([A-Z][a-z]+)', content):
        case_name = f"{match.group(1)} v. {match.group(2)}"
        if case_name not in seen:
            seen.add(case_name)
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 100)
            context = content[start:end].strip()
            yield ProbeTarget(identifier=case_name, source_file=source_file, context=context[:200])
            count += 1
            if count >= max_count:
                return

    # Legal citations (e.g., "123 F.3d 456", "456 U.S. 789")
    for match in re.finditer(r'\b(\d{1,4})\s+(F\.\d+d?|U\.S\.|S\.Ct\.|L\.Ed\.)\s+(\d+)', content):
        citation = match.group(0)
        if citation not in seen:
            seen.add(citation)
            yield ProbeTarget(identifier=citation, source_file=source_file, context=f"Citation: {citation}")
            count += 1
            if count >= max_count:
                return

    # Proper nouns (potential party names, judge names)
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', content):
        name = match.group(1)
        # Skip common words that look like names
        if name.lower() in {'the court', 'the state', 'united states', 'the defendant', 'the plaintiff'}:
            continue
        if name not in seen and len(name) > 5:
            seen.add(name)
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 100)
            context = content[start:end].strip()
            yield ProbeTarget(identifier=name, source_file=source_file, context=context[:200])
            count += 1
            if count >= max_count:
                return


def extract_general_identifiers(content: str, source_file: str, max_count: int = 20) -> Iterator[ProbeTarget]:
    """Extract general identifiers: proper nouns, organizations, dates."""
    seen = set()
    count = 0

    # Multi-word proper nouns (potential org names, people)
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', content):
        name = match.group(1)
        if name not in seen and len(name) > 5:
            seen.add(name)
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 100)
            context = content[start:end].strip()
            yield ProbeTarget(identifier=name, source_file=source_file, context=context[:200])
            count += 1
            if count >= max_count:
                return

    # Quoted strings (potential important phrases)
    for match in re.finditer(r'"([^"]{10,100})"', content):
        phrase = match.group(1)
        if phrase not in seen and not phrase.startswith('http'):
            seen.add(phrase)
            yield ProbeTarget(identifier=phrase, source_file=source_file, context=f"Quoted: {phrase[:100]}")
            count += 1
            if count >= max_count:
                return


def extract_from_content(content: str, content_type: str = 'general', max_count: int = 50) -> Iterator[ProbeTarget]:
    """Extract from raw pasted content."""
    if content_type == 'auto':
        # Simple heuristic: if it has code-like patterns, treat as code
        if re.search(r'\bdef\s+\w+\(|class\s+\w+:|function\s+\w+\(|const\s+\w+\s*=', content):
            content_type = 'code'
        elif re.search(r'\bv\.\s+\w+|\d+\s+F\.\d+d?\s+\d+', content):
            content_type = 'legal'
        else:
            content_type = 'general'

    if content_type == 'code':
        yield from extract_code_identifiers(content, 'pasted_content', max_count)
    elif content_type == 'legal':
        yield from extract_legal_identifiers(content, 'pasted_content', max_count)
    else:
        yield from extract_general_identifiers(content, 'pasted_content', max_count)
