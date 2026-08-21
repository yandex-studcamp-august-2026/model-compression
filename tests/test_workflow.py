import unittest
from pathlib import Path


class WorkflowContractTest(unittest.TestCase):
    def test_gpu_artifacts_keep_one_directory_per_experiment(self):
        workflow = Path(".github/workflows/benchmark.yml").read_text(encoding="utf-8")
        gpu_download = workflow.split("pattern: bundle-gpu-*", maxsplit=1)[1].split(
            "- name: Authenticate", maxsplit=1
        )[0]

        self.assertNotIn("merge-multiple: true", gpu_download)
        self.assertIn("path: bundles", gpu_download)

    def test_gpu_job_declares_expected_hardware(self):
        workflow = Path(".github/workflows/benchmark.yml").read_text(encoding="utf-8")
        runner = Path("scripts/run_remote_gpu.sh").read_text(encoding="utf-8")

        self.assertIn("EXPECTED_GPU_MODEL: ${{ vars.EXPECTED_GPU_MODEL }}", workflow)
        self.assertIn("EXPECTED_GPU_MODEL is required", runner)
        self.assertIn('!= *"${EXPECTED_GPU_MODEL}"*', runner)

    def test_v100_preflight_uses_supported_trtexec_probe(self):
        runner = Path("scripts/run_remote_gpu.sh").read_text(encoding="utf-8")

        self.assertIn("trtexec --help", runner)
        self.assertNotIn("trtexec --version", runner)

    def test_s3_job_uses_frozen_lock(self):
        workflow = Path(".github/workflows/benchmark.yml").read_text(encoding="utf-8")
        fetch_job = workflow.split("fetch-inputs:", maxsplit=1)[1].split(
            "export-onnx:", maxsplit=1
        )[0]

        self.assertIn("uv sync --project trusted --frozen --extra storage", fetch_job)
        self.assertIn("uv run --project trusted --frozen --no-sync", fetch_job)
        self.assertNotIn("pip install boto3", fetch_job)

    def test_jobs_do_not_resync_after_locked_install(self):
        workflow = Path(".github/workflows/benchmark.yml").read_text(encoding="utf-8")

        for line in workflow.splitlines():
            if "uv run" in line:
                self.assertIn("--frozen --no-sync", line)

    def test_one_failed_export_does_not_block_other_cpu_candidates(self):
        workflow = Path(".github/workflows/benchmark.yml").read_text(encoding="utf-8")
        cpu_job = workflow.split("benchmark-cpu:", maxsplit=1)[1].split(
            "benchmark-gpu:", maxsplit=1
        )[0]

        self.assertIn("needs.discover.result == 'success'", cpu_job)
        self.assertNotIn("needs.export-onnx.result == 'success'", cpu_job)


if __name__ == "__main__":
    unittest.main()
