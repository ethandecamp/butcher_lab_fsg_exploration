"""Guard tests for the container packaging: the invariants, not the daemon.

Nothing here runs Docker. That is on purpose -- a test that needs a daemon is a test
that fails in CI, on Dan's machine, and on every laptop where the daemon happens to be
stopped, and the acceptance evidence for an image that actually builds and runs lives
in TASK-017's Log where a human can read the pasted output.

What these tests do protect is the set of properties that are easy to break silently by
editing the ``Dockerfile`` and never noticing:

* the ~2.1 GB dataset is **mounted, never baked** -- the day someone "fixes" a missing
  file by copying ``one_way_fsg_model`` into the image, this fails;
* the **test suite runs during the build**, which is what makes a broken image
  impossible to produce;
* the default entrypoint is ``smoke``, the one that needs no network, no model and no
  key;
* ``ANTHROPIC_API_KEY`` is removed inside the container, so a key that leaked in
  through ``docker run -e`` cannot quietly turn a keyless run into a billed one;
* ``.dockerignore`` keeps the virtualenv and the generated reports out of the build
  context, while keeping the one recorded transcript ``smoke`` replays.

Run with::

    cd src && python3 -m unittest discover -s ../tests -t .. -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

#: The packaging lives beside ``src/`` and ``tests/``, not inside them.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = PACKAGE_ROOT / "Dockerfile"
DOCKERIGNORE = PACKAGE_ROOT / ".dockerignore"
INSTRUCTIONS = PACKAGE_ROOT / "INSTRUCTIONS_DOCKER.md"


def _read(path: Path) -> str:
    """Read a packaging file, failing with its path if it is missing.

    Args:
        path: The file to read.

    Returns:
        Its text.
    """
    if not path.is_file():
        raise AssertionError(f"{path} is missing; the container packaging is incomplete")
    return path.read_text(encoding="utf-8")


class DockerfileTests(unittest.TestCase):
    """The Dockerfile still encodes the decisions TASK-017 made."""

    def setUp(self) -> None:
        """Load the Dockerfile once per test."""
        self.text = _read(DOCKERFILE)
        self.instructions = [
            line.strip()
            for line in self.text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def test_the_dataset_is_never_copied_into_the_image(self):
        """The dataset is never copied into the image."""
        # The path may appear as a *mount point* and as the entrypoint's default root.
        # What must never appear is an instruction that brings its contents in.
        brought_in = [
            line
            for line in self.instructions
            if line.split()[0].upper() in {"COPY", "ADD"}
            and ("one_way_fsg_model" in line or "FSG Results" in line)
        ]
        self.assertEqual(brought_in, [], "the dataset is being baked into the image")

    def test_nothing_is_copied_from_outside_the_build_context(self):
        """Nothing is copied from outside the build context."""
        # `..` in a COPY is how the dataset would sneak in. Docker refuses it today,
        # but a future --build-context or a moved Dockerfile would not.
        for line in self.instructions:
            if line.split()[0].upper() in {"COPY", "ADD"}:
                self.assertNotIn("..", line)

    def test_the_dataset_arrives_as_a_mount(self):
        """The dataset arrives as a mount."""
        self.assertIn("/data/one_way_fsg_model", self.text)
        self.assertIn("VOLUME", self.text)

    def test_the_test_suite_runs_during_the_build(self):
        """The test suite runs during the build."""
        gate = [line for line in self.instructions if line.startswith("RUN") and "pytest" in line]
        self.assertTrue(gate, "no build-time RUN invokes pytest, so a broken image can be built")

    def test_the_default_command_is_smoke(self):
        """The default command is smoke."""
        self.assertIn('CMD ["smoke"]', self.text)

    def test_the_entrypoint_unsets_the_anthropic_key(self):
        """The entrypoint unsets the anthropic key."""
        self.assertIn("unset ANTHROPIC_API_KEY", self.text)

    def test_smoke_replays_the_recorded_transcript(self):
        """Smoke replays the recorded transcript."""
        self.assertIn("--generator transcript", self.text)
        self.assertIn("demo_transcript.json", self.text)

    def test_the_base_image_is_one_that_publishes_both_architectures(self):
        """The base image is one that publishes both architectures."""
        self.assertIn("FROM python:3.11-slim", self.text)

    def test_smoke_fails_loudly_when_the_dataset_is_not_mounted(self):
        """Smoke fails loudly when the dataset is not mounted."""
        self.assertIn("no FSG dataset at", self.text)


class DockerignoreTests(unittest.TestCase):
    """The build context stays small, and carries the one file smoke needs."""

    def setUp(self) -> None:
        """Load the .dockerignore once per test."""
        self.lines = [
            line.strip()
            for line in _read(DOCKERIGNORE).splitlines()
            if line.strip() and not line.startswith("#")
        ]

    def test_the_virtualenv_is_excluded(self):
        """The virtualenv is excluded."""
        self.assertIn(".venv/", self.lines)

    def test_generated_output_is_excluded(self):
        """Generated output is excluded."""
        self.assertIn("data/", self.lines)

    def test_caches_are_excluded(self):
        """Caches are excluded."""
        for pattern in ("__pycache__/", ".pytest_cache/", ".ruff_cache/"):
            self.assertIn(pattern, self.lines)

    def test_the_demo_transcript_is_re_included(self):
        """The demo transcript is re-included."""
        # Excluding all of data/ without this would make the smoke entrypoint -- the
        # one thing that proves the container works offline -- impossible to run.
        self.assertIn("!data/demo_transcript.json", self.lines)

    def test_the_transcript_smoke_replays_actually_exists(self):
        """The transcript smoke replays actually exists."""
        self.assertTrue((PACKAGE_ROOT / "data" / "demo_transcript.json").is_file())


class InstructionsTests(unittest.TestCase):
    """The instructions cover the steps a non-programmer cannot infer."""

    def setUp(self) -> None:
        """Load the instructions once per test."""
        self.text = _read(INSTRUCTIONS)

    def test_it_covers_the_read_only_mount(self):
        """It covers the read only mount."""
        self.assertIn(":ro", self.text)

    def test_it_covers_git_lfs(self):
        """It covers git lfs."""
        self.assertIn("git lfs", self.text)

    def test_it_has_a_failure_table(self):
        """It has a failure table."""
        self.assertIn("| Symptom |", self.text)

    def test_it_shows_the_offline_smoke_run(self):
        """It shows the offline smoke run."""
        self.assertIn("--network none", self.text)

    def test_it_warns_worktree_users_about_the_symlink(self):
        """It warns worktree users about the symlink."""
        # `lane-*/one_way_fsg_model` is a symlink, and a bind mount of a symlink is not
        # a bind mount of the directory it points at. Someone hitting that needs to be
        # told, not left to work it out.
        self.assertIn("symlink", self.text.lower())


if __name__ == "__main__":
    unittest.main()
