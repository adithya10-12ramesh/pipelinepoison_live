"""
harness/taxonomy.py

The frozen 6 x 4 attack taxonomy (Phase 2 deliverable). This module is the
single source of truth for goal and surface identifiers — payloads.json,
context_assembler.py, and judge.py all import from here so the taxonomy can
never silently drift between corpus and code.
"""

from enum import Enum


class Goal(str, Enum):
    EXFIL_SECRET = "EXFIL_SECRET"           # exfiltrate secrets/credentials via output
    MODIFY_CODE = "MODIFY_CODE"             # get the assistant to inject/alter code
    APPROVE_MAL_PR = "APPROVE_MAL_PR"       # get a malicious PR auto-approved
    DENY_LEGIT_PR = "DENY_LEGIT_PR"         # get a legitimate PR blocked/denied
    MISLEAD_TRIAGE = "MISLEAD_TRIAGE"       # cause mis-triage / mis-labelling
    AGENT_MISUSE = "AGENT_MISUSE"           # coerce misuse of agent tool-calls


class Surface(str, Enum):
    PR_DESCRIPTION = "PR_DESCRIPTION"
    COMMIT_MESSAGE = "COMMIT_MESSAGE"
    SOURCE_DIFF = "SOURCE_DIFF"
    CI_LOG_OUTPUT = "CI_LOG_OUTPUT"


GOALS = list(Goal)
SURFACES = list(Surface)


def class_id(goal: Goal | str, surface: Surface | str) -> str:
    g = goal.value if isinstance(goal, Goal) else goal
    s = surface.value if isinstance(surface, Surface) else surface
    return f"{g}__{s}"


ALL_CLASSES = [class_id(g, s) for g in GOALS for s in SURFACES]  # 24 classes

assert len(ALL_CLASSES) == 24, "Taxonomy must yield exactly 24 attack classes"
