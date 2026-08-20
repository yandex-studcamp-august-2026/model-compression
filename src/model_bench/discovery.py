from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from model_bench.quality import load_metrics, validate_conclusion


class Backend(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


MARKERS = {"CPU": Backend.CPU, "GPU": Backend.GPU}
TRIGGER_FILES = frozenset(
    {*MARKERS, "forward.py", "weights.url", "weights.sha256", "dataset.json"}
)
EXPERIMENT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
INFRASTRUCTURE_FILES = frozenset(
    {Path(".python-version"), Path("pyproject.toml"), Path("uv.lock")}
)
INFRASTRUCTURE_ROOTS = (Path(".github/workflows"), Path("src"), Path("scripts"))


@dataclass(frozen=True)
class Candidate:
    directory: Path
    marker: Path
    backend: Backend
    forward: Path
    weights_url: Path
    weights_sha256: Path
    dataset: Path
    metrics: Path
    conclusion: Path

    @property
    def name(self) -> str:
        return self.directory.name

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "directory": self.directory.as_posix(),
            "backend": self.backend.value,
        }


def candidate_from_directory(directory: Path) -> Candidate | None:
    markers = [directory / name for name in MARKERS if (directory / name).is_file()]
    if not markers:
        return None
    if len(markers) != 1:
        raise ValueError(
            f"Candidate {directory} must contain exactly one of CPU or GPU"
        )
    marker = markers[0]
    if marker.read_bytes():
        raise ValueError(f"Backend marker must be empty: {marker}")
    if EXPERIMENT_NAME.fullmatch(directory.name) is None:
        raise ValueError(f"Experiment directory must use snake_case: {directory.name}")
    forward = directory / "forward.py"
    if not forward.is_file():
        raise ValueError(f"Candidate {directory} has {marker.name} but no forward.py")
    weights_url = directory / "weights.url"
    if not weights_url.is_file() or not weights_url.read_text(encoding="utf-8").strip():
        raise ValueError(
            f"Candidate {directory} has {marker.name} but no non-empty weights.url"
        )
    weights_sha256 = directory / "weights.sha256"
    if (
        not weights_sha256.is_file()
        or re.fullmatch(r"[0-9a-f]{64}\n?", weights_sha256.read_text(encoding="ascii"))
        is None
    ):
        raise ValueError(
            f"Candidate {directory} must contain weights.sha256 with one "
            "lowercase SHA-256"
        )
    dataset = directory / "dataset.json"
    if not dataset.is_file() or dataset.is_symlink():
        raise ValueError(f"Candidate {directory} must contain dataset.json")
    try:
        dataset_value = json.loads(dataset.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("dataset.json must contain valid JSON") from exc
    if not isinstance(dataset_value, dict):
        raise ValueError("dataset.json must contain a JSON object")
    if set(dataset_value) != {"uri", "include"}:
        raise ValueError("dataset.json must contain exactly uri and include")
    if not isinstance(dataset_value["uri"], str) or not dataset_value["uri"].startswith(
        "s3://"
    ):
        raise ValueError("dataset.json uri must be an s3:// URI")
    includes = dataset_value["include"]
    if (
        not isinstance(includes, list)
        or not includes
        or any(not isinstance(value, str) or not value.strip() for value in includes)
    ):
        raise ValueError("dataset.json include must be a non-empty string list")
    metrics = directory / "metrics.json"
    conclusion = directory / "conclusion.md"
    load_metrics(metrics, directory.name)
    validate_conclusion(conclusion)
    return Candidate(
        directory,
        marker,
        MARKERS[marker.name],
        forward,
        weights_url,
        weights_sha256,
        dataset,
        metrics,
        conclusion,
    )


def discover(root: Path) -> list[Candidate]:
    if not root.exists():
        return []
    candidates = []
    directories = {
        marker.parent for name in MARKERS for marker in root.glob(f"*/{name}")
    }
    for directory in sorted(directories):
        candidate = candidate_from_directory(directory)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def changed_files(base: str, head: str) -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base, head, "--"],
        text=True,
        capture_output=True,
        check=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def discover_changed(root: Path, base: str, head: str) -> list[Candidate]:
    files = changed_files(base, head)
    directories = {
        path.parent
        for path in files
        if (path.name in TRIGGER_FILES or path.suffix == ".py")
        and path.is_relative_to(root)
        and path.parent.parent == root
    }
    candidates = []
    for directory in sorted(directories):
        candidate = candidate_from_directory(directory)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def reject_mixed_infrastructure_candidate_change(
    root: Path, base: str, head: str
) -> None:
    files = changed_files(base, head)
    candidate_change = any(
        path.is_relative_to(root) and path.parent.parent == root for path in files
    )
    infrastructure_change = any(
        path in INFRASTRUCTURE_FILES
        or any(path.is_relative_to(prefix) for prefix in INFRASTRUCTURE_ROOTS)
        for path in files
    )
    if candidate_change and infrastructure_change:
        raise ValueError(
            "Infrastructure and experiment changes must use separate pull requests"
        )


def candidates_json(candidates: list[Candidate], backend: Backend | None = None) -> str:
    selected = [
        item for item in candidates if backend is None or item.backend is backend
    ]
    return json.dumps([item.as_dict() for item in selected], ensure_ascii=False)


def discover_candidates(
    root: Path,
    *,
    base: str | None = None,
    head: str | None = None,
    explicit: Path | None = None,
    backend: Backend | None = None,
) -> list[Candidate]:
    if explicit is not None:
        candidate = candidate_from_directory(explicit)
        candidates = [candidate] if candidate is not None else []
    elif base is not None or head is not None:
        if base is None or head is None:
            raise ValueError("--base and --head must be provided together")
        candidates = discover_changed(root, base, head)
    else:
        candidates = discover(root)
    return [item for item in candidates if backend is None or item.backend is backend]
