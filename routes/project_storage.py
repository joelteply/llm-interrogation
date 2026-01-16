"""
Project storage - FACADE over repository.

This module exists for backwards compatibility.
New code should use: from repositories import get_repository

All functions delegate to the repository layer.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from config import PROJECTS_DIR
from repositories import get_repository
from models import Project, ProbeResponse


def get_project_dir(name: str) -> Path:
    """Get project directory path."""
    return PROJECTS_DIR / name


def project_exists(name: str) -> bool:
    """Check if project exists."""
    return get_repository().projects.exists(name)


def create_project(name: str, topic: str = "", selected_models: list = None) -> dict:
    """Create a new project."""
    project = Project(
        name=name,
        topic=topic,
        selected_models=selected_models or [],
    )
    get_repository().projects.save(project)

    # Ensure corpus file exists
    corpus_file = get_project_dir(name) / "corpus.jsonl"
    corpus_file.touch(exist_ok=True)

    return project.model_dump(mode="json")


def load_project_meta(name: str) -> Optional[dict]:
    """Load project metadata as dict (legacy interface)."""
    project = get_repository().projects.get(name)
    if not project:
        return None

    data = project.model_dump(mode="json")

    # Add legacy field aliases
    data["created"] = data.get("created_at")
    data["updated"] = data.get("updated_at")

    # Load embedded data that was in project.json
    skeptic = get_repository().projects.get_skeptic_feedback(name)
    if skeptic:
        data["skeptic_feedback"] = skeptic.model_dump(mode="json")

    verification = get_repository().projects.get_entity_verification(name)
    if verification:
        data["entity_verification"] = verification.model_dump(mode="json")

    return data


def save_project_meta(name: str, meta: dict) -> None:
    """Save project metadata from dict (legacy interface)."""
    repo = get_repository()

    # Handle embedded objects separately
    skeptic_data = meta.pop("skeptic_feedback", None)
    verification_data = meta.pop("entity_verification", None)
    web_leads_data = meta.pop("web_leads", None)

    # Handle legacy timestamp fields
    if "created" in meta and "created_at" not in meta:
        meta["created_at"] = meta.pop("created")
    if "updated" in meta:
        meta.pop("updated")  # Will be set by touch()

    # Validate and save project
    project = Project.model_validate(meta)
    repo.projects.save(project)

    # Save embedded objects
    if skeptic_data:
        from models import SkepticFeedback
        repo.projects.save_skeptic_feedback(name, SkepticFeedback.model_validate(skeptic_data))

    if verification_data:
        from models import EntityVerification
        repo.projects.save_entity_verification(name, EntityVerification.model_validate(verification_data))


def append_response(name: str, response: dict) -> None:
    """Append response to corpus."""
    resp = ProbeResponse.model_validate(response)
    get_repository().corpus.append(name, resp)


def load_corpus(name: str) -> list:
    """Load corpus as list of dicts (legacy interface)."""
    responses = get_repository().corpus.get_for_project(name)
    return [r.model_dump(mode="json") for r in responses]


def corpus_size(name: str) -> int:
    """Get corpus size."""
    return get_repository().corpus.count(name)


def append_narrative(name: str, narrative: str) -> None:
    """Append narrative to history."""
    import json
    project_dir = get_project_dir(name)
    narratives_file = project_dir / "narratives.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "narrative": narrative
    }

    with open(narratives_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_narratives(name: str) -> list:
    """Load narrative history."""
    import json
    project_dir = get_project_dir(name)
    narratives_file = project_dir / "narratives.jsonl"

    if not narratives_file.exists():
        return []

    narratives = []
    with open(narratives_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    narratives.append(json.loads(line))
                except:
                    pass
    return narratives


def delete_project(name: str) -> bool:
    """Delete project."""
    return get_repository().projects.delete(name)


def list_projects() -> list:
    """List all projects as dicts (legacy interface)."""
    repo = get_repository()
    projects = repo.projects.list()

    result = []
    for p in projects:
        data = p.model_dump(mode="json")
        data["created"] = data.get("created_at")
        data["updated"] = data.get("updated_at")
        data["corpus_count"] = repo.corpus.count(p.name)
        result.append(data)

    return result


def migrate_all_old_projects() -> int:
    """
    Migrate old single-file projects to directory structure.

    The repository pattern now handles legacy formats via Pydantic's
    flexible validation (extra="allow"/"ignore"). This function checks
    for any truly old formats and migrates them.

    Returns number of projects migrated.
    """
    import json

    migrated = 0

    # Check for old single-file format (project_name.json in PROJECTS_DIR root)
    for item in PROJECTS_DIR.iterdir():
        if item.is_file() and item.suffix == ".json" and item.stem != "config":
            # Old format: projects/myproject.json
            # New format: projects/myproject/project.json
            project_name = item.stem
            new_dir = PROJECTS_DIR / project_name

            if not new_dir.exists():
                new_dir.mkdir(parents=True)

                # Move the JSON file
                new_path = new_dir / "project.json"

                try:
                    with open(item) as f:
                        data = json.load(f)

                    # Ensure name field exists
                    data["name"] = project_name

                    with open(new_path, "w") as f:
                        json.dump(data, f, indent=2, default=str)

                    # Remove old file
                    item.unlink()
                    migrated += 1
                    print(f"[MIGRATE] {project_name}: single file -> directory")

                except Exception as e:
                    print(f"[MIGRATE] Failed to migrate {project_name}: {e}")

    return migrated
