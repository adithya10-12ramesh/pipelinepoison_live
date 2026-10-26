"""
live_pipeline/live_context.py

Thin wrapper around the GitHub REST API (via PyGithub) for the live
pipeline: reading a real PR's surfaces (title, body, diff, and a tracked
"CI log" file that stands in for the CI_LOG_OUTPUT taxonomy surface), and
posting the review bot's actions back to GitHub.

Kept separate from bot.py so open_prs.py, bot.py, and fetch_live_results.py
all read/write PRs through the same functions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from github import Github

CI_LOG_PATH = "ci/last_run.log"  # tracked file in the sandbox repo


def get_client() -> Github:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("LIVE_GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Set GITHUB_TOKEN (provided automatically inside GitHub Actions) "
            "or LIVE_GITHUB_TOKEN (when running locally, e.g. from open_prs.py)."
        )
    return Github(token)


def get_repo(gh: Github | None = None):
    gh = gh or get_client()
    full_name = os.environ.get("GITHUB_REPOSITORY") or os.environ.get("LIVE_GITHUB_REPO")
    if not full_name:
        raise RuntimeError(
            "Set GITHUB_REPOSITORY (provided automatically inside GitHub Actions) "
            "or LIVE_GITHUB_REPO=owner/repo (when running locally)."
        )
    return gh.get_repo(full_name)


@dataclass
class LivePRSurfaces:
    pr_number: int
    title: str
    body: str
    diff: str
    commit_messages: str
    ci_log: str
    mitigation_state: str
    labels: list[str]


def mitigation_state_from_labels(pr) -> str:
    """open_prs.py encodes which mitigation state a PR represents as a
    `mitigation:<state>` label, since there's no per-PR workflow_dispatch
    input for a `pull_request`-triggered run."""
    for label in pr.get_labels():
        if label.name.startswith("mitigation:"):
            return label.name.split(":", 1)[1]
    return "none"


def read_surfaces(pr, repo) -> LivePRSurfaces:
    diff_text = "\n".join((f.patch or "") for f in pr.get_files())
    commit_messages = "\n".join(c.commit.message for c in pr.get_commits())
    try:
        ci_log_file = repo.get_contents(CI_LOG_PATH, ref=pr.head.sha)
        ci_log = ci_log_file.decoded_content.decode("utf-8", errors="replace")
    except Exception:
        ci_log = ""
    return LivePRSurfaces(
        pr_number=pr.number,
        title=pr.title or "",
        body=pr.body or "",
        diff=diff_text,
        commit_messages=commit_messages,
        ci_log=ci_log,
        mitigation_state=mitigation_state_from_labels(pr),
        labels=[l.name for l in pr.get_labels()],
    )


# --------------------------------------------------------------------------- #
# Actions -- what the bot is allowed to actually do to the PR. Every call
# here is a REAL GitHub API call; bot.py decides whether to invoke these at
# all (that decision is the output-validator mitigation gate).
# --------------------------------------------------------------------------- #

def post_comment(pr, body: str) -> int:
    return pr.create_issue_comment(body).id


def submit_review(pr, event: str, body: str) -> int:
    """event: 'APPROVE' | 'REQUEST_CHANGES' | 'COMMENT'"""
    return pr.create_review(body=body, event=event).id


def add_label(repo, pr, label_name: str) -> None:
    try:
        repo.get_label(label_name)
    except Exception:
        repo.create_label(name=label_name, color="ededed")
    pr.add_to_labels(label_name)


def ensure_label(repo, name: str, color: str = "c5def5") -> None:
    try:
        repo.get_label(name)
    except Exception:
        repo.create_label(name=name, color=color)
