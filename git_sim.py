"""
TEMPORARY PLACEHOLDER — NOT THE REAL FILE.

The original git_sim.py was accidentally deleted by a background agent on
2026-08-31 (an untracked file, never committed, so there is no git history
to restore it from). This stub exists only so mastery_server.py can import
successfully and the rest of the Mastery Hub works again — every route that
actually depends on git_sim (the /git-sim page and its /api/git-sim/*
endpoints) is intentionally non-functional here and says so plainly rather
than faking a working git sandbox.

If you have a copy of the real git_sim.py anywhere (another machine, an
open editor tab, a backup), replace this file with that one. Otherwise
this needs to be rebuilt from scratch — see git-scenarios-cheatsheet.md
for the write-up the original scenarios paired with, which may help
reconstruct the scenario list.
"""

SCENARIOS = []
SCENARIOS_BY_ID = {}

_UNAVAILABLE_MSG = (
    "The git simulator's source file (git_sim.py) was accidentally lost and hasn't "
    "been restored yet. This feature is temporarily unavailable — nothing else in "
    "the hub is affected."
)


def exists(box_id: str) -> bool:
    return False


def read_state(box_id: str) -> dict:
    return {"scenario": None, "teammate_log": []}


def write_state(box_id: str, state: dict) -> None:
    pass


def get_graph(box_id: str) -> dict:
    return {"you": _UNAVAILABLE_MSG, "remote": "", "status": "", "config": ""}


def reset_sandbox(box_id: str) -> None:
    pass


def run_user_command(box_id: str, command: str) -> dict:
    return {"ok": False, "output": _UNAVAILABLE_MSG}


def write_file(box_id: str, content: str) -> None:
    pass


def read_file(box_id: str) -> str:
    return ""


def apply_merge_strategy(box_id: str, strategy: str) -> dict:
    return {"you": _UNAVAILABLE_MSG, "remote": "", "status": "", "config": ""}
