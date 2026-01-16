"""
JSON file backend - stores data as JSON/JSONL files.

Directory structure:
    projects/{name}/
        project.json      - Project metadata
        corpus.jsonl      - Responses (append-only)
        research.jsonl    - Research documents
        findings.json     - Aggregated findings
"""

import json
import threading
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime

from config import PROJECTS_DIR
from models import (
    Project,
    ProbeResponse,
    Findings,
    ResearchDocument,
    SkepticFeedback,
    EntityVerification,
)
from .base import (
    Repository,
    ProjectRepository,
    CorpusRepository,
    ResearchRepository,
)


class WriteQueue:
    """Thread-safe write serialization."""

    def __init__(self):
        self._lock = threading.Lock()

    def write_json(self, path: Path, data: dict) -> None:
        """Atomic JSON write."""
        with self._lock:
            temp = path.with_suffix(".json.tmp")
            with open(temp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            temp.replace(path)

    def append_jsonl(self, path: Path, data: dict) -> None:
        """Append to JSONL file."""
        with self._lock:
            with open(path, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")


_write_queue = WriteQueue()


class JsonProjectRepository(ProjectRepository):
    """JSON file implementation of project repository."""

    def __init__(self, base_path: Path = None):
        self._base_path = base_path or PROJECTS_DIR

    def _project_dir(self, name: str) -> Path:
        return self._base_path / name

    def _project_file(self, name: str) -> Path:
        return self._project_dir(name) / "project.json"

    def get(self, id: str) -> Optional[Project]:
        path = self._project_file(id)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[WARN] Corrupt project.json for {id}: {e}")
            return None

        # Handle legacy field names
        if "created" in data and "created_at" not in data:
            data["created_at"] = data.pop("created")
        if "updated" in data and "updated_at" not in data:
            data["updated_at"] = data.pop("updated")

        return Project.model_validate(data)

    def save(self, entity: Project) -> None:
        project_dir = self._project_dir(entity.name)
        project_dir.mkdir(parents=True, exist_ok=True)

        entity.touch()
        data = entity.model_dump(mode="json")

        _write_queue.write_json(self._project_file(entity.name), data)

    def delete(self, id: str) -> bool:
        import shutil
        project_dir = self._project_dir(id)
        if not project_dir.exists():
            return False
        shutil.rmtree(project_dir)
        return True

    def list(self) -> list[Project]:
        self._base_path.mkdir(exist_ok=True)
        projects = []

        for d in self._base_path.iterdir():
            if d.is_dir():
                project = self.get(d.name)
                if project:
                    projects.append(project)

        return sorted(projects, key=lambda p: p.updated_at, reverse=True)

    def exists(self, id: str) -> bool:
        return self._project_file(id).exists()

    # Extended methods

    def get_findings(self, project_id: str) -> Optional[Findings]:
        path = self._project_dir(project_id) / "findings.json"
        if not path.exists():
            return None
        with open(path) as f:
            return Findings.model_validate(json.load(f))

    def save_findings(self, project_id: str, findings: Findings) -> None:
        path = self._project_dir(project_id) / "findings.json"
        _write_queue.write_json(path, findings.model_dump(mode="json"))

    def get_skeptic_feedback(self, project_id: str) -> Optional[SkepticFeedback]:
        project = self.get(project_id)
        if not project:
            return None

        # Load from project metadata (legacy location)
        path = self._project_file(project_id)
        with open(path) as f:
            data = json.load(f)

        if "skeptic_feedback" not in data:
            return None

        return SkepticFeedback.model_validate(data["skeptic_feedback"])

    def save_skeptic_feedback(self, project_id: str, feedback: SkepticFeedback) -> None:
        project = self.get(project_id)
        if not project:
            return

        # Store in project metadata
        path = self._project_file(project_id)
        with open(path) as f:
            data = json.load(f)

        data["skeptic_feedback"] = feedback.model_dump(mode="json")
        data["updated"] = datetime.now().isoformat()

        _write_queue.write_json(path, data)

    def get_entity_verification(self, project_id: str) -> Optional[EntityVerification]:
        path = self._project_file(project_id)
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        if "entity_verification" not in data:
            return None

        return EntityVerification.model_validate(data["entity_verification"])

    def save_entity_verification(self, project_id: str, verification: EntityVerification) -> None:
        path = self._project_file(project_id)
        if not path.exists():
            return

        with open(path) as f:
            data = json.load(f)

        data["entity_verification"] = verification.model_dump(mode="json")
        data["updated"] = datetime.now().isoformat()

        _write_queue.write_json(path, data)


class JsonCorpusRepository(CorpusRepository):
    """JSON file implementation of corpus repository."""

    def __init__(self, base_path: Path = None):
        self._base_path = base_path or PROJECTS_DIR

    def _corpus_file(self, project_id: str) -> Path:
        return self._base_path / project_id / "corpus.jsonl"

    def get(self, id: str) -> Optional[ProbeResponse]:
        # Corpus entries don't have individual IDs in current impl
        raise NotImplementedError("Use get_for_project or iterate")

    def save(self, entity: ProbeResponse) -> None:
        raise NotImplementedError("Use append")

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Corpus is append-only")

    def list(self) -> list[ProbeResponse]:
        raise NotImplementedError("Use get_for_project")

    def exists(self, id: str) -> bool:
        raise NotImplementedError()

    def append(self, project_id: str, response: ProbeResponse) -> None:
        path = self._corpus_file(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_queue.append_jsonl(path, response.model_dump(mode="json"))

    def get_for_project(self, project_id: str) -> list[ProbeResponse]:
        return list(self.iterate(project_id))

    def count(self, project_id: str) -> int:
        path = self._corpus_file(project_id)
        if not path.exists():
            return 0

        count = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def iterate(self, project_id: str) -> Iterator[ProbeResponse]:
        path = self._corpus_file(project_id)
        if not path.exists():
            return

        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield ProbeResponse.model_validate(data)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"[WARN] Corrupt line {line_num} in {project_id}/corpus.jsonl: {e}")


class JsonResearchRepository(ResearchRepository):
    """JSON file implementation of research repository."""

    def __init__(self, base_path: Path = None):
        self._base_path = base_path or PROJECTS_DIR

    def _research_file(self, project_id: str) -> Path:
        return self._base_path / project_id / "research.jsonl"

    def get(self, id: str) -> Optional[ResearchDocument]:
        raise NotImplementedError("Use get_for_project")

    def save(self, entity: ResearchDocument) -> None:
        raise NotImplementedError("Use append")

    def delete(self, id: str) -> bool:
        raise NotImplementedError()

    def list(self) -> list[ResearchDocument]:
        raise NotImplementedError("Use get_for_project")

    def exists(self, id: str) -> bool:
        raise NotImplementedError()

    def get_for_project(self, project_id: str) -> list[ResearchDocument]:
        path = self._research_file(project_id)
        if not path.exists():
            return []

        docs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(ResearchDocument.model_validate(json.loads(line)))
                except Exception:
                    pass
        return docs

    def append(self, project_id: str, doc: ResearchDocument) -> bool:
        # Check for duplicate
        existing = self.get_for_project(project_id)
        if any(d.url == doc.url for d in existing):
            return False

        path = self._research_file(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_queue.append_jsonl(path, doc.model_dump(mode="json"))
        return True

    def query(self, project_id: str, terms: list[str], limit: int = 5) -> list[ResearchDocument]:
        docs = self.get_for_project(project_id)
        terms_lower = [t.lower() for t in terms]

        scored = []
        for doc in docs:
            if not doc.is_useful:
                continue
            content_lower = doc.content.lower()
            score = sum(1 for t in terms_lower if t in content_lower)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:limit]]


class JsonRepository(Repository):
    """JSON file backend implementation."""

    def __init__(self, base_path: Path = None):
        self._base_path = base_path or PROJECTS_DIR
        self._projects = JsonProjectRepository(self._base_path)
        self._corpus = JsonCorpusRepository(self._base_path)
        self._research = JsonResearchRepository(self._base_path)

    @property
    def projects(self) -> ProjectRepository:
        return self._projects

    @property
    def corpus(self) -> CorpusRepository:
        return self._corpus

    @property
    def research(self) -> ResearchRepository:
        return self._research
