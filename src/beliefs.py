"""Bayesian belief tracker over goal door identity.

Each agent maintains P(goal=d) for each door.  The LLM extracts structured
evidence from observations — which doors a signal supports and an estimated
reliability — then a standard Bayesian update adjusts the posterior.

Fully optional: toggled by TrialConfig.enable_bayesian.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

EVIDENCE_PROMPT = """\
You are analyzing signals in a door-finding task.

Given the observation below, extract evidence about which door might be the goal.
For EACH signal you see, output exactly one line in this format:

EVIDENCE: <door_label_or_NONE> | <reliability 0.0-1.0>

- door_label: the door it supports (e.g. "red arched door"), or NONE if unclear
- reliability: 1.0 = certainly true, 0.0 = certainly false, 0.5 = unsure
- Distractors often contradict each other, use vague language, or sound alarmist

Observation:
{observation}

Known doors: {door_list}"""


@dataclass
class BeliefState:
    """Probability distribution over which door is the goal."""

    door_ids: list[int] = field(default_factory=list)
    door_labels: dict[int, str] = field(default_factory=dict)
    beliefs: dict[int, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.door_ids and not self.beliefs:
            n = len(self.door_ids)
            self.beliefs = {d: 1.0 / n for d in self.door_ids}

    def update_from_evidence(self, supported_door: int | None, reliability: float) -> None:
        if supported_door is None or reliability < 0.1:
            return
        for d in self.door_ids:
            if d == supported_door:
                self.beliefs[d] *= (1.0 + reliability)
            else:
                self.beliefs[d] *= max(0.01, 1.0 - reliability * 0.3)
        self._normalize()
        self.history.append({
            "supported": supported_door,
            "reliability": reliability,
            "posterior": dict(self.beliefs),
        })

    def _normalize(self) -> None:
        total = sum(self.beliefs.values())
        if total > 0:
            for d in self.beliefs:
                self.beliefs[d] /= total

    def entropy(self) -> float:
        h = 0.0
        for p in self.beliefs.values():
            if p > 1e-12:
                h -= p * math.log2(p)
        return h

    def max_belief(self) -> float:
        return max(self.beliefs.values()) if self.beliefs else 0.0

    def map_door(self) -> int | None:
        if not self.beliefs:
            return None
        return max(self.beliefs, key=self.beliefs.get)

    def map_label(self) -> str:
        d = self.map_door()
        return self.door_labels.get(d, f"door {d}") if d is not None else "unknown"

    def belief_on_true_goal(self, goal_id: int) -> float:
        return self.beliefs.get(goal_id, 0.0)

    def to_text(self) -> str:
        sorted_doors = sorted(self.beliefs.items(), key=lambda x: -x[1])
        lines = []
        for did, prob in sorted_doors:
            label = self.door_labels.get(did, f"door {did}")
            bar = "#" * int(prob * 20)
            lines.append(f"  {label}: {prob:.1%} {bar}")
        return "\n".join(lines)

    def to_prior_text(self) -> str:
        sorted_doors = sorted(self.beliefs.items(), key=lambda x: -x[1])
        parts = []
        for did, prob in sorted_doors[:3]:
            label = self.door_labels.get(did, f"door {did}")
            parts.append(f"{label} ({prob:.0%})")
        return "Belief ranking: " + " > ".join(parts)


def extract_evidence(
    llm,
    observation_text: str,
    door_labels: dict[int, str],
) -> list[tuple[int | None, float]]:
    """Use LLM to extract structured evidence from an observation."""
    door_list = ", ".join(f"{label} (node {did})" for did, label in door_labels.items())
    prompt = EVIDENCE_PROMPT.format(observation=observation_text, door_list=door_list)
    raw = ""
    for attempt in range(3):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
            if raw is None:
                raise ValueError("None response")
            break
        except Exception:
            if attempt == 2:
                return []
            time.sleep(2 * (attempt + 1))

    evidence: list[tuple[int | None, float]] = []
    label_to_id = {v.lower(): k for k, v in door_labels.items()}

    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line.upper().startswith("EVIDENCE:"):
            continue
        rest = line[9:].strip()
        parts = [p.strip() for p in rest.split("|")]
        if len(parts) < 2:
            continue
        try:
            door_part = parts[0].strip().strip('"').lower()
            rel_part = parts[1].strip()

            door_id = None
            if door_part and door_part != "none":
                for label, did in label_to_id.items():
                    if label in door_part or door_part in label:
                        door_id = did
                        break
                if door_id is None:
                    for label, did in label_to_id.items():
                        color = label.split()[0]
                        if color in door_part:
                            door_id = did
                            break

            rel_match = re.search(r"[\d.]+", rel_part)
            reliability = float(rel_match.group()) if rel_match else 0.5
            reliability = max(0.0, min(1.0, reliability))
            evidence.append((door_id, reliability))
        except (ValueError, IndexError):
            continue

    return evidence
