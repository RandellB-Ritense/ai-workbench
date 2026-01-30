"""
Project management utilities for AI Workbench.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import json


@dataclass(frozen=True)
class ProjectInfo:
    name: str
    root: Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config_dir: Path
    output_dir: Path
    vector_store_dir: Path
    job_dir: Path
    chat_sessions_dir: Path
    logs_dir: Path


def build_project_paths(project_root: Path) -> ProjectPaths:
    """Build standard project paths under the project root."""
    root = project_root.resolve()
    return ProjectPaths(
        root=root,
        config_dir=root / "config",
        output_dir=root / "outputs",
        vector_store_dir=root / "vector-stores",
        job_dir=root / "jobs",
        chat_sessions_dir=root / "chat-sessions",
        logs_dir=root / "logs",
    )


def ensure_project_dirs(paths: ProjectPaths) -> None:
    """Ensure all project directories exist."""
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.vector_store_dir.mkdir(parents=True, exist_ok=True)
    paths.job_dir.mkdir(parents=True, exist_ok=True)
    paths.chat_sessions_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)


def write_project_metadata(project: ProjectInfo, paths: ProjectPaths) -> Path:
    """Write project metadata to the project's config directory."""
    metadata = {
        "name": project.name,
        "root": str(paths.root),
        "created_at": datetime.now().isoformat(),
    }
    metadata_path = paths.config_dir / "project.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def serialize_project_state(project: ProjectInfo) -> Dict[str, str]:
    """Serialize project info for Gradio state."""
    return {"name": project.name, "root": str(project.root.resolve())}


def project_paths_from_state(state: Optional[Dict[str, str]]) -> ProjectPaths:
    """Parse project paths from a serialized project state."""
    if not state or "root" not in state:
        raise ValueError("No active project.")
    return build_project_paths(Path(state["root"]))


def load_project_from_dir(project_root: Path) -> ProjectInfo:
    """Load and validate a project from its root directory."""
    root = project_root.resolve()
    config_dir = root / "config"
    metadata_path = config_dir / "project.json"

    if not metadata_path.exists():
        raise ValueError("Invalid project: missing config/project.json")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    name = metadata.get("name")
    stored_root = metadata.get("root")

    if not name or not stored_root:
        raise ValueError("Invalid project: missing name or root in project.json")

    if Path(stored_root).resolve() != root:
        raise ValueError("Invalid project: root mismatch in project.json")

    return ProjectInfo(name=str(name), root=root)
