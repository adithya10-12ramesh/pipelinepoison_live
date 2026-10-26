#!/usr/bin/env python3
"""
live_pipeline/fetch_benign_prs.py

Samples real, already-merged PRs from permissively-licensed open-source
repos to build the benign-PR control set (step 4 of "making it real
world": attack success rate alone says nothing about your mitigations'
false-positive rate on legitimate traffic).

Writes fixtures/benign_prs.json in the format open_prs.py --include-benign
expects. Keep this list small (20-30) and the diffs short -- these just
need to look like ordinary, unremarkable PRs; they don't need to be
interesting, and nothing here is opened as a PR yet (open_prs.py does
that against YOUR sandbox repo, not the source repos).
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()
from live_pipeline.live_context import get_client

PERMISSIVE_LICENSES = ("mit", "apache-2.0", "bsd-3-clause")


def sample_benign_prs(gh, n: int, query_repos: list[str]) -> list[dict]:
    out = []
    per_repo_target = max(1, n // max(1, len(query_repos)))
    for repo_name in query_repos:
        if len(out) >= n:
            break
        repo = gh.get_repo(repo_name)
        if repo.license is None or repo.license.key not in PERMISSIVE_LICENSES:
            print(f"  skipping {repo_name}: license is "
                  f"{repo.license.key if repo.license else 'none'}, not permissive")
            continue
        count = 0
        for pr in repo.get_pulls(state="closed", sort="updated", direction="desc"):
            if count >= per_repo_target or len(out) >= n:
                break
            if not pr.merged:
                continue
            files = list(pr.get_files())
            if not files or sum(f.changes for f in files) > 40:
                continue  # keep it small and unremarkable
            snippet = "\n".join((f.patch or "")[:500] for f in files[:2])
            out.append({
                "id": f"benign_{repo_name.replace('/', '_')}_{pr.number}",
                "title": pr.title,
                "body": (pr.body or "")[:500],
                "diff_marker_content": snippet or "# benign change\n",
                "true_severity": "low",
                "source_repo": repo_name,
                "source_pr": pr.number,
            })
            count += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument(
        "--repos", nargs="+",
        default=["pallets/flask", "psf/requests", "pytest-dev/pytest"],
        help="permissively-licensed repos to sample merged PRs from",
    )
    parser.add_argument("--out", default="fixtures/benign_prs.json")
    args = parser.parse_args()

    gh = get_client()
    prs = sample_benign_prs(gh, args.count, args.repos)
    Path(args.out).write_text(json.dumps(prs, indent=2), encoding="utf-8")
    print(f"[fetch_benign_prs] wrote {len(prs)} benign PRs to {args.out}")


if __name__ == "__main__":
    main()
