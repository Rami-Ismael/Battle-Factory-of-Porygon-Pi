"""Tests for the guard that stops tools/sync.py from undoing edits merged on GitHub.

Run from the repository root:
    python3 -m unittest discover -s tests -v

Each test builds throwaway git repositories: the two laptop folders (vault and
code), a bare repository standing in for GitHub, and a clone of it holding a copy
of tools/sync.py. setUp runs one real sync, so every test starts from a baseline;
a test then lands a GitHub-side edit by pushing it from the clone.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SYNC = Path(__file__).resolve().parents[1] / "tools" / "sync.py"
EDIT = "edited on GitHub\n"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def listed(output):
    """The indented (action, path) rows in the sync's output."""
    return {tuple(line.split(None, 1)) for line in output.splitlines() if line.startswith("  ")}


class SyncGuard(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.vault, self.code, self.clone = root / "vault", root / "code", root / "clone"
        self.allow = root / "allow.txt"
        # Keep the user's git config (signing, hooks, default branch) out of these repositories.
        self.env = dict(os.environ, BF_VAULT=str(self.vault), BF_CODE=str(self.code),
                        GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
                        GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.com",
                        GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.com")
        self.git(root, "init", "-q", "--bare", "-b", "main", "github.git")
        for repo in (self.vault, self.code, self.clone):
            self.git(root, "init", "-q", "-b", "main", repo.name)
        write(self.vault / "plan.md", "laptop plan\n")
        write(self.vault / "ideas.md", "ideas\n")
        write(self.code / "run.py", "print('run')\n")
        write(self.clone / "tools" / "sync.py", SYNC.read_text())
        self.git(self.clone, "add", "tools")
        self.git(self.clone, "commit", "-q", "-m", "Add the sync script")
        self.git(self.clone, "remote", "add", "origin", str(root / "github.git"))
        self.git(self.clone, "push", "-q", "-u", "origin", "main")
        first = self.sync()
        self.assertEqual(first.returncode, 0, first.stderr)

    def git(self, cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, env=self.env, check=True,
                              capture_output=True, text=True).stdout

    def sync(self, *args):
        return subprocess.run([sys.executable, "tools/sync.py", *args], cwd=self.clone,
                              env=self.env, capture_output=True, text=True)

    def merge_on_github(self, path, text=None):
        """Push one commit from the clone, as a merged pull request lands; text=None deletes."""
        if text is None:
            (self.clone / path).unlink()
        else:
            write(self.clone / path, text)
        self.git(self.clone, "add", "-A", "notes", "code")
        self.git(self.clone, "commit", "-q", "-m", f"Edit {path} on GitHub (#1)")
        self.git(self.clone, "push", "-q")

    def github_head(self):
        return self.git(self.clone, "log", "-1", "--format=%H %s", "origin/main").strip()

    def test_refuses_to_overwrite_a_github_edit(self):
        self.merge_on_github("notes/plan.md", EDIT)
        before = self.github_head()
        result = self.sync()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(listed(result.stderr), {("overwrite", "notes/plan.md")})
        self.assertEqual((self.clone / "notes/plan.md").read_text(), EDIT)
        self.assertEqual(self.github_head(), before)

    def test_dry_run_prints_the_same_refusal_and_writes_nothing(self):
        self.merge_on_github("notes/plan.md", EDIT)
        before = self.git(self.clone, "rev-parse", "HEAD")
        dry, real = self.sync("--dry-run"), self.sync("--no-push")
        self.assertEqual(dry.returncode, 1)
        self.assertEqual(dry.stderr, real.stderr)
        forced = self.sync("--dry-run", "--force")
        self.assertEqual(forced.returncode, 0, forced.stderr)
        self.assertIn(("overwrite", "notes/plan.md"), listed(forced.stdout))
        self.assertEqual(self.git(self.clone, "status", "--porcelain"), "")
        self.assertEqual(self.git(self.clone, "rev-parse", "HEAD"), before)

    def test_force_overwrites_and_becomes_the_next_baseline(self):
        self.merge_on_github("notes/plan.md", EDIT)
        result = self.sync("--force")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.clone / "notes/plan.md").read_text(), "laptop plan\n")
        self.assertTrue(self.github_head().split(" ", 1)[1].startswith("sync from laptop"))
        again = self.sync("--dry-run")
        self.assertEqual((again.returncode, again.stdout), (0, "already in sync\n"))

    def test_allow_file_overwrites_only_the_listed_paths(self):
        self.merge_on_github("notes/plan.md", EDIT)
        self.merge_on_github("code/run.py", "print('github')\n")
        self.allow.write_text("notes/plan.md\n")
        result = self.sync("--dry-run", "--allow", str(self.allow))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(listed(result.stderr), {("overwrite", "code/run.py")})
        self.allow.write_text("notes/plan.md\ncode/run.py\n")
        result = self.sync("--allow", str(self.allow))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.clone / "code/run.py").read_text(), "print('run')\n")

    def test_files_added_or_deleted_on_github_are_guarded_too(self):
        self.merge_on_github("notes/new.md", "added on GitHub\n")
        self.merge_on_github("notes/ideas.md")
        result = self.sync("--dry-run")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(listed(result.stderr), {("delete", "notes/new.md"), ("re-add", "notes/ideas.md")})

    def test_syncs_once_the_laptop_has_the_github_version(self):
        self.merge_on_github("notes/plan.md", EDIT)
        write(self.vault / "plan.md", EDIT)  # copied back from GitHub
        (self.vault / "ideas.md").unlink()  # laptop-only changes still go through
        write(self.code / "run.py", "print('laptop')\n")
        result = self.sync()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.clone / "notes/ideas.md").exists())
        self.assertEqual((self.clone / "code/run.py").read_text(), "print('laptop')\n")

    def test_stops_before_pushing_when_github_changes_during_the_sync(self):
        root = self.clone.parent
        pr = root / "pr"
        self.git(root, "clone", "-q", str(root / "github.git"), "pr")
        write(pr / "notes/ideas.md", EDIT)
        self.git(pr, "commit", "-q", "-am", "Edit ideas.md on GitHub mid-sync (#2)")
        # The clone's post-commit hook lands that pull request between the guard's check and the push.
        hook = self.clone / ".git" / "hooks" / "post-commit"
        write(hook, f'#!/bin/sh\nunset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE\ngit -C "{pr}" push -q origin main\n')
        hook.chmod(0o755)
        write(self.vault / "plan.md", "laptop edit\n")  # gives the sync something to commit
        result = self.sync()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(listed(result.stderr), {("overwrite", "notes/ideas.md")})
        self.assertEqual(self.git(root / "github.git", "log", "-1", "--format=%s", "main").strip(),
                         "Edit ideas.md on GitHub mid-sync (#2)")
        hook.unlink()
        again = self.sync("--dry-run")  # the pull request stays above the baseline
        self.assertEqual(listed(again.stderr), {("overwrite", "notes/ideas.md")})


if __name__ == "__main__":
    unittest.main()
