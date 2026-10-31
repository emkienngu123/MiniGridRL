"""Prepare local release assets; never publishes or changes remote repositories."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args, capture=False):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                          capture_output=capture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-results", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "submission" / "manifest.json").read_text(encoding="utf-8"))
    branch = git("branch", "--show-current", capture=True).stdout.strip()
    if args.commit_results:
        if manifest["status"] != "full_matrix":
            raise ValueError("Final result commit requires the complete experiment matrix")
        if branch != "assignment/exploration":
            raise ValueError("Artifacts are ready; switch to assignment/exploration before committing results")
        paths = ["checkpoints", "results/full", "submission"]
        git("add", "--", *paths)
        changed = git("diff", "--cached", "--name-only", "--", *paths, capture=True).stdout.strip()
        if changed:
            git("commit", "--only", "-m", "Add full exploration experiment results and submission artifacts", "--", *paths)
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    bundle = dist / "MiniGridExplorer-source.bundle"
    git("bundle", "create", str(bundle), "--all")
    files = [bundle, dist / "MiniGridExplorer-Windows-CPU.zip",
             ROOT / "submission" / "technical-report.pdf", ROOT / "submission" / "demo.mp4"]
    entries = []
    checksums = []
    for path in files:
        if not path.is_file():
            raise ValueError(f"Missing release artifact: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest})
        checksums.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    release = {"target_repository": "emkienngu123/minigrid-explorer", "published": False,
               "evidence_status": manifest["status"], "branch": branch,
               "source_commit": git("rev-parse", "HEAD", capture=True).stdout.strip(), "assets": entries}
    (dist / "release-manifest.json").write_text(json.dumps(release, indent=2), encoding="utf-8")
    (dist / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="ascii")
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
