"""
Schema-driven format handling.

Single source of truth for LLM output formats.
"""

import re
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Section:
    key: str
    label: str
    type: str  # single_line, paragraph, numbered_list, bullet_list
    required: bool
    instruction: str
    item_format: str = ""
    min_items: int = 0
    allowed_values: list = field(default_factory=list)


@dataclass
class FormatSchema:
    name: str
    description: str
    sections: list[Section]

    def to_prompt_instructions(self) -> str:
        """Generate prompt format instructions from schema."""
        lines = ["OUTPUT FORMAT:"]

        for section in self.sections:
            if section.type == "single_line":
                lines.append(f"{section.label}: [{section.instruction}]")
            elif section.type == "paragraph":
                lines.append(f"{section.label}: [{section.instruction}]")
            elif section.type == "numbered_list":
                lines.append(f"\n{section.label} ({section.instruction}):")
                lines.append(f"1. {section.item_format}")
                lines.append(f"2. [Continue listing...]")
            elif section.type == "bullet_list":
                lines.append(f"\n{section.label}:")
                lines.append(f"* {section.item_format}")

            if section.type in ("single_line", "paragraph"):
                lines.append("")  # blank line after

        return "\n".join(lines)

    def parse(self, text: str) -> dict:
        """Parse LLM output according to schema."""
        result = {}

        for section in self.sections:
            value = self._extract_section(text, section)
            result[section.key] = value

        return result

    def _extract_section(self, text: str, section: Section):
        """Extract a section's content from text."""
        label_pattern = re.escape(section.label)

        if section.type == "single_line":
            match = re.search(rf'{label_pattern}:?\s*\n?([^\n]+)', text, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('[]')
            return ""

        elif section.type == "paragraph":
            # Find this section and capture until next section label
            next_labels = [s.label for s in self.sections if s.label != section.label]
            next_pattern = "|".join(re.escape(l) for l in next_labels)
            match = re.search(
                rf'{label_pattern}:?\s*\n?([\s\S]*?)(?={next_pattern}:|$)',
                text, re.IGNORECASE
            )
            if match:
                return match.group(1).strip()
            return ""

        elif section.type in ("numbered_list", "bullet_list"):
            # Find section start
            start_match = re.search(rf'{label_pattern}[^:]*:', text, re.IGNORECASE)
            if not start_match:
                return []

            # Find next section
            next_labels = [s.label for s in self.sections if s.label != section.label]
            remaining = text[start_match.end():]

            # Find where next section starts
            next_start = len(remaining)
            for label in next_labels:
                match = re.search(rf'\n{re.escape(label)}[^:]*:', remaining, re.IGNORECASE)
                if match and match.start() < next_start:
                    next_start = match.start()

            section_text = remaining[:next_start].strip()

            # Parse list items
            items = []
            if section.type == "numbered_list":
                items = re.findall(r'^\d+\.\s*(.+)$', section_text, re.MULTILINE)
            else:
                items = re.findall(r'^[*\-]\s*(.+)$', section_text, re.MULTILINE)

            return items

        return None

    def to_json_schema(self) -> dict:
        """Export as JSON schema for frontend."""
        return {
            "name": self.name,
            "description": self.description,
            "sections": [
                {
                    "key": s.key,
                    "label": s.label,
                    "type": s.type,
                    "required": s.required,
                }
                for s in self.sections
            ]
        }


def load_schema(name: str) -> FormatSchema:
    """Load a schema by name."""
    schema_dir = Path(__file__).parent
    schema_file = schema_dir / f"{name}.yaml"

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema not found: {name}")

    with open(schema_file) as f:
        data = yaml.safe_load(f)

    sections = [
        Section(
            key=s["key"],
            label=s["label"],
            type=s["type"],
            required=s.get("required", False),
            instruction=s.get("instruction", ""),
            item_format=s.get("item_format", ""),
            min_items=s.get("min_items", 0),
            allowed_values=s.get("allowed_values", []),
        )
        for s in data["sections"]
    ]

    return FormatSchema(
        name=data["name"],
        description=data["description"],
        sections=sections,
    )


# Convenience: pre-load narrative format
def get_narrative_schema() -> FormatSchema:
    return load_schema("narrative_format")
