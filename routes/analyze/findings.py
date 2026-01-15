"""
Findings Accumulator - Persist and build on discoveries.

Don't lose specificity. Every detail matters.
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

FINDINGS_DIR = Path(__file__).parent.parent.parent / "projects"


@dataclass
class Finding:
    """A specific finding with full detail."""
    entity: str
    entity_type: str  # PERSON, EMAIL, ORGANIZATION, LOCATION, etc.
    context: str  # Where/how it was found
    confidence: float
    sources: list[str] = field(default_factory=list)  # Which models/docs
    related_to: list[str] = field(default_factory=list)  # Connected entities
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    notes: str = ""


@dataclass
class Theory:
    """A working theory built from findings."""
    summary: str
    supporting_findings: list[str]  # Finding entity names
    confidence: float
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "active"  # active, confirmed, disproven


class FindingsAccumulator:
    """
    Accumulate findings over time. Never lose detail.

    Persists to disk so findings survive across sessions.
    """

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.findings_path = FINDINGS_DIR / project_name / "_findings.json"
        self.findings: dict[str, Finding] = {}
        self.theories: list[Theory] = []
        self._load()

    def _load(self):
        """Load existing findings from disk."""
        if self.findings_path.exists():
            try:
                with open(self.findings_path) as f:
                    data = json.load(f)

                for key, val in data.get('findings', {}).items():
                    self.findings[key] = Finding(**val)

                for t in data.get('theories', []):
                    self.theories.append(Theory(**t))

                print(f"[FINDINGS] Loaded {len(self.findings)} findings, {len(self.theories)} theories")
            except Exception as e:
                print(f"[FINDINGS] Error loading: {e}")

    def _save(self):
        """Save findings to disk."""
        self.findings_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'findings': {k: asdict(v) for k, v in self.findings.items()},
            'theories': [asdict(t) for t in self.theories],
            'last_updated': datetime.utcnow().isoformat()
        }

        with open(self.findings_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_finding(self,
                    entity: str,
                    entity_type: str,
                    context: str,
                    confidence: float,
                    source: str = None,
                    related_to: list[str] = None):
        """
        Add or update a finding. Accumulates, never overwrites.
        """
        key = entity.lower().strip()

        if key in self.findings:
            # Update existing - accumulate sources and context
            existing = self.findings[key]
            if source and source not in existing.sources:
                existing.sources.append(source)
            if context and context not in existing.context:
                existing.context += f"\n---\n{context}"
            if related_to:
                for r in related_to:
                    if r not in existing.related_to:
                        existing.related_to.append(r)
            # Update confidence (weighted average toward new)
            existing.confidence = (existing.confidence * 0.7) + (confidence * 0.3)
        else:
            # New finding
            self.findings[key] = Finding(
                entity=entity,
                entity_type=entity_type,
                context=context,
                confidence=confidence,
                sources=[source] if source else [],
                related_to=related_to or []
            )

        self._save()
        return self.findings[key]

    def add_theory(self, summary: str, supporting: list[str], confidence: float):
        """Add a new theory."""
        theory = Theory(
            summary=summary,
            supporting_findings=supporting,
            confidence=confidence
        )
        self.theories.append(theory)
        self._save()
        return theory

    def update_theory(self, index: int, **updates):
        """Update an existing theory."""
        if 0 <= index < len(self.theories):
            theory = self.theories[index]
            for key, val in updates.items():
                if hasattr(theory, key):
                    setattr(theory, key, val)
            theory.updated = datetime.utcnow().isoformat()
            self._save()

    def get_findings_by_type(self, entity_type: str) -> list[Finding]:
        """Get all findings of a specific type."""
        return [f for f in self.findings.values()
                if f.entity_type.upper() == entity_type.upper()]

    def get_related(self, entity: str) -> list[Finding]:
        """Get findings related to an entity."""
        key = entity.lower().strip()
        related = []
        for f in self.findings.values():
            if key in [r.lower() for r in f.related_to]:
                related.append(f)
            if f.entity.lower() == key:
                # Get findings this one is related to
                for r in f.related_to:
                    if r.lower() in self.findings:
                        related.append(self.findings[r.lower()])
        return related

    def build_narrative(self) -> str:
        """Build a detailed narrative from all findings. Never abbreviate."""
        if not self.findings:
            return "No findings yet."

        lines = ["## Accumulated Findings\n"]

        # Group by type
        by_type = {}
        for f in self.findings.values():
            t = f.entity_type
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(f)

        for entity_type, findings in sorted(by_type.items()):
            lines.append(f"\n### {entity_type}S\n")
            for f in sorted(findings, key=lambda x: -x.confidence):
                lines.append(f"**{f.entity}** ({f.confidence:.0%} confidence)")
                lines.append(f"  - Context: {f.context[:200]}...")
                if f.sources:
                    lines.append(f"  - Sources: {', '.join(f.sources)}")
                if f.related_to:
                    lines.append(f"  - Related to: {', '.join(f.related_to)}")
                lines.append("")

        # Add theories
        if self.theories:
            lines.append("\n### Working Theories\n")
            for i, t in enumerate(self.theories):
                status_marker = "✓" if t.status == "confirmed" else "✗" if t.status == "disproven" else "→"
                lines.append(f"{status_marker} **Theory {i+1}** ({t.confidence:.0%}): {t.summary}")
                lines.append(f"  Supporting: {', '.join(t.supporting_findings)}")
                lines.append("")

        return "\n".join(lines)

    def export_full(self) -> dict:
        """Export everything - never lose detail."""
        return {
            'project': self.project_name,
            'findings_count': len(self.findings),
            'theories_count': len(self.theories),
            'findings': {k: asdict(v) for k, v in self.findings.items()},
            'theories': [asdict(t) for t in self.theories],
            'narrative': self.build_narrative()
        }


def get_project_findings(project_name: str) -> FindingsAccumulator:
    """Get or create findings accumulator for a project."""
    return FindingsAccumulator(project_name)
