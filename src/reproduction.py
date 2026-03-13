"""Agent reproduction: prior compression, birth, lineage tracking.

Supports multiple reproduction triggers:
  - periodic: after N interactions
  - on_success: agent finds the goal, births a child before retiring
  - on_context_overflow: when context buffer overflows (not yet implemented)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .agent import Agent
from .config import TrialConfig


_counter = itertools.count(1)


def next_id() -> str:
    return f"agent_{next(_counter)}"


def reset_id_counter() -> None:
    global _counter
    _counter = itertools.count(1)


@dataclass
class BirthEvent:
    parent_id: str
    child_id: str
    generation: int
    step: int
    trigger: str  # "periodic", "on_success", "on_context_overflow"
    parent_prior: str
    child_prior: str
    parent_context_snapshot: str


@dataclass
class Lineage:
    parents: dict[str, str | None] = field(default_factory=dict)
    generations: dict[str, int] = field(default_factory=dict)
    births: list[BirthEvent] = field(default_factory=list)

    def add_root(self, agent_id: str) -> None:
        self.parents[agent_id] = None
        self.generations[agent_id] = 0

    def add_birth(self, event: BirthEvent) -> None:
        self.parents[event.child_id] = event.parent_id
        self.generations[event.child_id] = event.generation
        self.births.append(event)

    def get_depth(self, agent_id: str) -> int:
        return self.generations.get(agent_id, 0)

    def max_generation(self) -> int:
        return max(self.generations.values()) if self.generations else 0

    def tree_str(self, successful: set[str] | None = None) -> str:
        successful = successful or set()
        roots = [a for a, p in self.parents.items() if p is None]
        lines: list[str] = []

        def visit(aid: str, prefix: str, is_last: bool) -> None:
            marker = "└── " if is_last else "├── "
            tag = " [SUCCESS]" if aid in successful else ""
            gen = self.generations.get(aid, "?")
            trigger = ""
            for b in self.births:
                if b.child_id == aid:
                    trigger = f" ({b.trigger})"
                    break
            lines.append(f"{prefix}{marker}{aid} (gen {gen}){trigger}{tag}")
            children = [c for c, p in self.parents.items() if p == aid]
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, c in enumerate(children):
                visit(c, new_prefix, i == len(children) - 1)

        for i, r in enumerate(roots):
            visit(r, "", i == len(roots) - 1)
        return "\n".join(lines)


def birth_agent(
    parent: Agent,
    prior: str,
    config: TrialConfig,
    reasoning_llm=None,
    utility_llm=None,
) -> Agent:
    """Create a child agent inheriting a compressed prior."""
    aid = next_id()
    parent.children_birthed += 1
    return Agent(
        agent_id=aid,
        config=config,
        prior=prior if config.inherit_prior else "",
        parent_id=parent.agent_id,
        generation=parent.generation + 1,
        reasoning_llm=reasoning_llm or parent.reasoning_llm,
        utility_llm=utility_llm or parent.utility_llm,
    )
