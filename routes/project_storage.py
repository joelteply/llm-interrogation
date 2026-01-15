"""
Project storage layer - directory-based structure.

projects/{name}/
  project.json      # metadata only (topic, models, settings)
  corpus.jsonl      # responses, one JSON per line (append-only)
  narratives.jsonl  # narratives, one per line
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from config import PROJECTS_DIR


def get_project_dir(name: str) -> Path:
    """Get project directory path."""
    return PROJECTS_DIR / name


def project_exists(name: str) -> bool:
    """Check if project exists."""
    project_dir = get_project_dir(name)
    return project_dir.exists() and (project_dir / "project.json").exists()


def create_project(name: str, topic: str = "", selected_models: list = None) -> dict:
    """Create a new project directory structure."""
    project_dir = get_project_dir(name)
    project_dir.mkdir(parents=True, exist_ok=True)

    project = {
        "name": name,
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "topic": topic,
        "angles": [],
        "selected_models": selected_models or [],
        "hidden_entities": [],
        "promoted_entities": [],
        "user_notes": "",
        "narrative": "",
        "narrative_updated": None,
    }

    with open(project_dir / "project.json", 'w') as f:
        json.dump(project, f, indent=2)

    # Create empty corpus and narratives files
    (project_dir / "corpus.jsonl").touch()
    (project_dir / "narratives.jsonl").touch()

    return project


def load_project_meta(name: str) -> Optional[dict]:
    """Load project metadata only (fast)."""
    project_dir = get_project_dir(name)
    meta_file = project_dir / "project.json"

    if not meta_file.exists():
        return None

    with open(meta_file) as f:
        return json.load(f)


def save_project_meta(name: str, meta: dict) -> None:
    """Save project metadata."""
    project_dir = get_project_dir(name)
    meta["updated"] = datetime.now().isoformat()

    with open(project_dir / "project.json", 'w') as f:
        json.dump(meta, f, indent=2)


def append_response(name: str, response: dict) -> None:
    """Append a single response to corpus (thread-safe append)."""
    project_dir = get_project_dir(name)
    corpus_file = project_dir / "corpus.jsonl"

    # Single line append - atomic on most filesystems
    line = json.dumps(response) + "\n"
    with open(corpus_file, 'a') as f:
        f.write(line)


def load_corpus(name: str) -> list:
    """Load all responses from corpus."""
    project_dir = get_project_dir(name)
    corpus_file = project_dir / "corpus.jsonl"

    if not corpus_file.exists():
        return []

    responses = []
    with open(corpus_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Corrupt line {line_num} in {name}/corpus.jsonl: {e}")
                # Skip corrupt lines instead of failing entire load

    return responses


def corpus_size(name: str) -> int:
    """Get corpus size without loading all data."""
    project_dir = get_project_dir(name)
    corpus_file = project_dir / "corpus.jsonl"

    if not corpus_file.exists():
        return 0

    count = 0
    with open(corpus_file) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def append_narrative(name: str, narrative: str) -> None:
    """Append a narrative to history."""
    project_dir = get_project_dir(name)
    narratives_file = project_dir / "narratives.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "narrative": narrative
    }

    line = json.dumps(entry) + "\n"
    with open(narratives_file, 'a') as f:
        f.write(line)


def load_narratives(name: str) -> list:
    """Load all narratives."""
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
                except json.JSONDecodeError:
                    pass

    return narratives


def delete_project(name: str) -> bool:
    """Delete a project directory."""
    import shutil
    project_dir = get_project_dir(name)

    if not project_dir.exists():
        return False

    shutil.rmtree(project_dir)
    return True


def list_projects() -> list:
    """List all projects (metadata only)."""
    PROJECTS_DIR.mkdir(exist_ok=True)
    projects = []

    for project_dir in PROJECTS_DIR.iterdir():
        if project_dir.is_dir():
            meta = load_project_meta(project_dir.name)
            if meta:
                # Add corpus count without loading all data
                meta["corpus_count"] = corpus_size(project_dir.name)
                projects.append(meta)

    return sorted(projects, key=lambda x: x.get('updated', ''), reverse=True)


def migrate_old_project(old_file: Path) -> bool:
    """Migrate old single-file project to new directory structure."""
    try:
        with open(old_file) as f:
            old_data = json.load(f)

        name = old_data.get("name", old_file.stem)
        project_dir = get_project_dir(name)

        if project_dir.exists():
            print(f"[MIGRATE] Skipping {name} - already migrated")
            return False

        project_dir.mkdir(parents=True, exist_ok=True)

        # Extract metadata
        meta = {
            "name": name,
            "created": old_data.get("created", datetime.now().isoformat()),
            "updated": old_data.get("updated", datetime.now().isoformat()),
            "topic": old_data.get("topic", ""),
            "angles": old_data.get("angles", []),
            "selected_models": old_data.get("selected_models", []),
            "hidden_entities": old_data.get("hidden_entities", []),
            "promoted_entities": old_data.get("promoted_entities", []),
            "user_notes": old_data.get("user_notes", ""),
            "narrative": old_data.get("narrative", ""),
            "narrative_updated": old_data.get("narrative_updated"),
            "questions": old_data.get("questions", []),
        }

        with open(project_dir / "project.json", 'w') as f:
            json.dump(meta, f, indent=2)

        # Write corpus as JSONL
        corpus = old_data.get("probe_corpus", [])
        with open(project_dir / "corpus.jsonl", 'w') as f:
            for response in corpus:
                f.write(json.dumps(response) + "\n")

        # Write narratives as JSONL
        narratives = old_data.get("narratives", [])
        with open(project_dir / "narratives.jsonl", 'w') as f:
            for n in narratives:
                if isinstance(n, dict):
                    f.write(json.dumps(n) + "\n")
                else:
                    f.write(json.dumps({"narrative": n, "timestamp": None}) + "\n")

        print(f"[MIGRATE] Migrated {name}: {len(corpus)} responses, {len(narratives)} narratives")

        # Rename old file to .bak
        old_file.rename(old_file.with_suffix('.json.bak'))

        return True
    except Exception as e:
        print(f"[MIGRATE] Failed to migrate {old_file}: {e}")
        return False


def migrate_all_old_projects() -> int:
    """Migrate all old single-file projects."""
    PROJECTS_DIR.mkdir(exist_ok=True)
    migrated = 0

    for old_file in PROJECTS_DIR.glob("*.json"):
        if migrate_old_project(old_file):
            migrated += 1

    return migrated
