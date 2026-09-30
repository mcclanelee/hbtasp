"""Kernel-owned mandatory-job terminal-state accounting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


TERMINAL_STATES = ("PRE_REJECT", "WAIT_EXPIRE", "LATE_COMPLETE", "ON_TIME")


@dataclass
class MandatoryTerminalRegistry:
    released_uids: set[str]
    states: dict[str, str | None] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.released_uids) == 0:
            raise ValueError("mandatory release set is empty")
        self.states = {uid: None for uid in self.released_uids}

    @classmethod
    def from_release_batches(cls, batches: Iterable[Iterable[object]]):
        uids: list[str] = []
        for jobs in batches:
            uids.extend(job.uid for job in jobs if job.mandatory)
        if len(uids) != len(set(uids)):
            raise RuntimeError("duplicate mandatory UID in release stream")
        return cls(set(uids))

    def transition(self, uid: str, state: str) -> str:
        if state not in TERMINAL_STATES:
            raise ValueError(f"unknown terminal state: {state}")
        if uid not in self.states:
            raise RuntimeError(f"terminal transition for unreleased mandatory UID: {uid}")
        if self.states[uid] is not None:
            raise RuntimeError(
                f"duplicate terminal transition for {uid}: {self.states[uid]} -> {state}"
            )
        self.states[uid] = state
        return state

    def finalize(self) -> dict[str, object]:
        missing = sorted(uid for uid, state in self.states.items() if state is None)
        if missing:
            raise RuntimeError(f"mandatory jobs without terminal state: {missing[:5]}")
        counts = Counter(self.states.values())
        released = len(self.states)
        total = sum(counts.values())
        if total != released:
            raise RuntimeError(f"terminal accounting mismatch: {total} != {released}")
        admitted = released - counts["PRE_REJECT"]
        admitted_failures = counts["WAIT_EXPIRE"] + counts["LATE_COMPLETE"]
        service_failures = counts["PRE_REJECT"] + admitted_failures
        canonical = sorted(self.states.items())
        digest = hashlib.sha256(
            json.dumps(canonical, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "mandatory_released": released,
            "mandatory_unique_terminal_uids": total,
            "mandatory_pre_reject": counts["PRE_REJECT"],
            "mandatory_wait_expire": counts["WAIT_EXPIRE"],
            "mandatory_late_complete": counts["LATE_COMPLETE"],
            "mandatory_on_time": counts["ON_TIME"],
            "mandatory_terminal_residual": released - total,
            "mandatory_pre_rejection_rate": counts["PRE_REJECT"] / released,
            "mandatory_admitted_dmr": admitted_failures / admitted if admitted else 0.0,
            "mandatory_service_failure_rate": service_failures / released,
            "mandatory_terminal_table_sha256": digest,
        }
