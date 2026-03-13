"""Voyager-inspired shared skill/convention library with categories.

Agents contribute verified conventions about the environment, organized
by type: navigation, signal_interpretation, door_patterns, spatial.
Other agents can query relevant conventions before decisions.
Usage is tracked so we can measure whether conventions actually help.
"""

from __future__ import annotations

from dataclasses import dataclass, field


CATEGORIES = ["navigation", "signal_interpretation", "door_patterns", "spatial", "general"]


@dataclass
class Convention:
    text: str
    category: str  # one of CATEGORIES
    added_by: str
    generation: int
    verified_count: int = 1
    times_queried: int = 0
    step_added: int = 0


class SkillLibrary:
    """Shared convention store for all agents in a trial."""

    def __init__(self, max_size: int = 50):
        self.conventions: list[Convention] = []
        self.max_size = max_size
        self._query_log: list[dict] = []  # for analysis

    def add(self, text: str, agent_id: str, generation: int, step: int) -> bool:
        text = text.strip()
        if not text or len(text) < 10:
            return False

        category = self._categorize(text)

        text_words = set(text.lower().split())
        for existing in self.conventions:
            existing_words = set(existing.text.lower().split())
            overlap = len(text_words & existing_words) / max(len(text_words | existing_words), 1)
            if overlap > 0.6:
                existing.verified_count += 1
                return False

        if len(self.conventions) >= self.max_size:
            self.conventions.sort(key=lambda c: c.verified_count + c.times_queried * 0.5)
            self.conventions.pop(0)

        self.conventions.append(Convention(
            text=text,
            category=category,
            added_by=agent_id,
            generation=generation,
            step_added=step,
        ))
        return True

    def _categorize(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["move", "navigate", "path", "edge", "center", "hub", "connectivity"]):
            return "navigation"
        if any(w in t for w in ["signal", "hint", "distractor", "trust", "misleading", "reliable"]):
            return "signal_interpretation"
        if any(w in t for w in ["door", "color", "red", "blue", "green", "yellow", "silver", "black"]):
            return "door_patterns"
        if any(w in t for w in ["region", "upper", "lower", "left", "right", "quadrant"]):
            return "spatial"
        return "general"

    def query(self, observation_text: str, max_results: int = 3) -> str:
        if not self.conventions:
            return ""
        obs_words = set(observation_text.lower().split())
        scored = []
        for conv in self.conventions:
            conv_words = set(conv.text.lower().split())
            keyword_score = len(obs_words & conv_words)
            # Boost verified and frequently-queried conventions
            quality_score = conv.verified_count * 0.2 + conv.times_queried * 0.05
            scored.append((keyword_score + quality_score, conv))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = []
        for score, conv in scored[:max_results]:
            if score > 0:
                conv.times_queried += 1
                top.append(f"[{conv.category}] {conv.text}")

        if not top:
            by_verified = sorted(self.conventions, key=lambda c: c.verified_count, reverse=True)
            for c in by_verified[:max_results]:
                c.times_queried += 1
                top.append(f"[{c.category}] {c.text}")

        result = "\n".join(f"- {t}" for t in top)
        return result

    def summary(self) -> str:
        if not self.conventions:
            return "(empty library)"
        lines = []
        for cat in CATEGORIES:
            cat_convs = [c for c in self.conventions if c.category == cat]
            if cat_convs:
                lines.append(f"\n  [{cat.upper()}]")
                for c in sorted(cat_convs, key=lambda c: c.verified_count, reverse=True):
                    lines.append(
                        f"    verified={c.verified_count} queried={c.times_queried} "
                        f"gen={c.generation}: {c.text}"
                    )
        return "\n".join(lines)

    def size(self) -> int:
        return len(self.conventions)

    def to_dicts(self) -> list[dict]:
        return [
            {
                "text": c.text,
                "category": c.category,
                "added_by": c.added_by,
                "generation": c.generation,
                "verified_count": c.verified_count,
                "times_queried": c.times_queried,
                "step_added": c.step_added,
            }
            for c in self.conventions
        ]

    def stats(self) -> dict:
        """Aggregate statistics for experiment reports."""
        if not self.conventions:
            return {"size": 0, "categories": {}, "total_queries": 0}
        cats = {}
        for c in self.conventions:
            cats.setdefault(c.category, 0)
            cats[c.category] += 1
        return {
            "size": len(self.conventions),
            "categories": cats,
            "total_queries": sum(c.times_queried for c in self.conventions),
            "avg_verified": sum(c.verified_count for c in self.conventions) / len(self.conventions),
        }
