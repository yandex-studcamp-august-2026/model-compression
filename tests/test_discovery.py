import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from model_bench.discovery import (
    Backend,
    candidate_from_directory,
    discover,
    discover_changed,
    reject_mixed_infrastructure_candidate_change,
)


def make_candidate(root: Path, name: str = "candidate", marker: str = "CPU") -> Path:
    directory = root / "experiments" / name
    directory.mkdir(parents=True)
    (directory / marker).write_bytes(b"")
    (directory / "forward.py").write_text("# model contract\n", encoding="utf-8")
    (directory / "weights.url").write_text("s3://bucket/model.pt\n", encoding="utf-8")
    (directory / "weights.sha256").write_text("a" * 64 + "\n", encoding="ascii")
    (directory / "dataset.json").write_text(
        json.dumps(
            {
                "uri": "s3://datasets/cityscapes",
                "include": ["images/val", "gtFine/val"],
            }
        ),
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "experiment_name": name,
                "author": "tester",
                "hypothesis": "candidate improves the baseline",
                "task": "segmentation",
                "recipe": "candidate",
                "metrics": {"mIoU": 0.61},
                "baseline_metrics": {"mIoU": 0.60},
            }
        ),
        encoding="utf-8",
    )
    (directory / "conclusion.md").write_text(
        "# Result\n\nValidated.\n", encoding="utf-8"
    )
    return directory


def commit(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=root, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


class DiscoveryTest(unittest.TestCase):
    def test_discovers_cpu_and_gpu_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cpu = make_candidate(root, "cpu_model")
            gpu = make_candidate(root, "gpu_model", "GPU")

            candidates = discover(root / "experiments")

            self.assertEqual([item.directory for item in candidates], [cpu, gpu])
            self.assertEqual(
                [item.backend for item in candidates], [Backend.CPU, Backend.GPU]
            )

    def test_marker_must_be_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = make_candidate(Path(temp))
            (directory / "CPU").write_text("run me", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be empty"):
                candidate_from_directory(directory)

    def test_exactly_one_marker_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = make_candidate(Path(temp))
            (directory / "GPU").write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "exactly one"):
                candidate_from_directory(directory)

    def test_weights_url_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = make_candidate(Path(temp))
            (directory / "weights.url").unlink()

            with self.assertRaisesRegex(ValueError, "weights.url"):
                candidate_from_directory(directory)

    def test_weights_checksum_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = make_candidate(Path(temp))
            (directory / "weights.sha256").write_text(
                "not-a-digest\n", encoding="ascii"
            )

            with self.assertRaisesRegex(ValueError, "weights.sha256"):
                candidate_from_directory(directory)

    def test_dataset_contract_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = make_candidate(Path(temp))
            (directory / "dataset.json").unlink()

            with self.assertRaisesRegex(ValueError, "dataset.json"):
                candidate_from_directory(directory)

    def test_experiment_name_must_be_snake_case(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = make_candidate(Path(temp), "Invalid-Name")

            with self.assertRaisesRegex(ValueError, "snake_case"):
                candidate_from_directory(directory)

    def test_conclusion_only_commit_does_not_select_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_candidate(root)
            self._init_repository(root)
            base = commit(root, "candidate")
            (root / "experiments/candidate/conclusion.md").write_text(
                "Looks good\n", encoding="utf-8"
            )
            head = commit(root, "conclusion")

            self.assertEqual(self._changed(root, base, head), [])

    def test_forward_change_selects_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = make_candidate(root)
            self._init_repository(root)
            base = commit(root, "candidate")
            (directory / "forward.py").write_text(
                "# updated contract\n", encoding="utf-8"
            )
            head = commit(root, "update forward")

            candidates = self._changed(root, base, head)

            self.assertEqual(
                [item.directory for item in candidates],
                [Path("experiments/candidate")],
            )

    def test_sibling_model_module_change_selects_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = make_candidate(root)
            (directory / "model.py").write_text("# architecture v1\n", encoding="utf-8")
            self._init_repository(root)
            base = commit(root, "candidate")
            (directory / "model.py").write_text("# architecture v2\n", encoding="utf-8")
            head = commit(root, "update architecture")

            candidates = self._changed(root, base, head)

            self.assertEqual(
                [item.directory for item in candidates],
                [Path("experiments/candidate")],
            )

    def test_cpu_to_gpu_promotion_selects_candidate_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = make_candidate(root)
            self._init_repository(root)
            base = commit(root, "cpu candidate")
            (directory / "CPU").unlink()
            (directory / "GPU").write_bytes(b"")
            head = commit(root, "promote to gpu")

            candidates = self._changed(root, base, head)

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].backend, Backend.GPU)

    def test_rejects_mixed_pipeline_and_candidate_pull_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = make_candidate(root)
            (root / "src/model_bench").mkdir(parents=True)
            (root / "src/model_bench/runtime.py").write_text("# v1\n", encoding="utf-8")
            self._init_repository(root)
            base = commit(root, "base")
            (directory / "forward.py").write_text("# v2\n", encoding="utf-8")
            (root / "src/model_bench/runtime.py").write_text("# v2\n", encoding="utf-8")
            head = commit(root, "mixed change")

            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                with self.assertRaisesRegex(ValueError, "separate pull requests"):
                    reject_mixed_infrastructure_candidate_change(
                        Path("experiments"), base, head
                    )
            finally:
                os.chdir(old_cwd)

    @staticmethod
    def _init_repository(root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "ci@example.test"],
            cwd=root,
            check=True,
        )
        subprocess.run(["git", "config", "user.name", "CI"], cwd=root, check=True)

    @staticmethod
    def _changed(root: Path, base: str, head: str):
        old_cwd = Path.cwd()
        try:
            os.chdir(root)
            return discover_changed(Path("experiments"), base, head)
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
