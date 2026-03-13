"""Graph-based POMDP environment with colored doors and structured signals.

A Random Geometric Graph where some nodes are doors (one is the goal).
Signals include spatial, color, relational, narrative, and pattern types.
Hints consistently point toward the true goal; distractors contradict each
other and point to different wrong answers — agents must learn to detect
consistency vs. contradiction.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

from .config import TrialConfig

# ---------------------------------------------------------------------------
# Door theming
# ---------------------------------------------------------------------------

DOOR_COLORS = [
    "red", "blue", "green", "yellow", "silver", "black", "white", "purple",
    "bronze", "crimson",
]
DOOR_SHAPES = [
    "arched", "narrow", "wide", "iron-banded", "ornate", "plain",
    "heavy", "wooden", "carved", "mosaic",
]

DOOR_DESCRIPTIONS = [
    "A tall {color} {shape} door with a {detail}.",
    "A weathered {color} {shape} door. {flavor}",
    "A {color} {shape} door with {detail}. {flavor}",
    "An imposing {color} {shape} door. {flavor}",
]

DOOR_DETAILS = [
    "brass handle tarnished with age",
    "small square window at eye level",
    "deep scratches along the frame",
    "freshly painted trim",
    "faded symbols carved into the wood",
    "heavy iron lock",
    "keyhole shaped like a star",
    "thin crack running down the center",
    "a worn bronze knocker",
    "faint runes etched into the lintel",
]

DOOR_FLAVORS = [
    "Cobwebs cover the hinges.",
    "The paint is peeling in long strips.",
    "It hums faintly when you touch it.",
    "Warm light leaks from underneath.",
    "Cold air seeps through the cracks.",
    "Someone has chalked a tally on the wall beside it.",
    "A faint smell of smoke comes from behind it.",
    "The floor in front of it is worn smooth.",
    "Faded footprints lead to and from it.",
    "A small bell hangs above the frame.",
]

# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------

REGIONS = ["upper-left", "upper-right", "lower-left", "lower-right", "center"]


def _pos_to_region(x: float, y: float) -> str:
    if 0.3 < x < 0.7 and 0.3 < y < 0.7:
        return "center"
    if y >= 0.5:
        return "upper-left" if x < 0.5 else "upper-right"
    return "lower-left" if x < 0.5 else "lower-right"


# ---------------------------------------------------------------------------
# Signal templates
# ---------------------------------------------------------------------------

# HINTS — all consistently reference the TRUE goal info
SPATIAL_HINTS = [
    "The goal door is in the {region} of the map.",
    "Head toward the {region} area to find the correct door.",
    "Reliable sources confirm the goal is in the {region}.",
    "An old map pinned to the wall shows an arrow pointing {region}.",
    "Worn footprints in the dust lead toward the {region} region.",
    "A compass rose painted on the floor points toward {region}.",
]

COLOR_HINTS = [
    "The correct door is {goal_color}.",
    "Look for the {goal_color} door — that's the one.",
    "A note reads: 'Trust the {goal_color} door.'",
    "Travelers who returned say the {goal_color} door leads to the goal.",
    "An inscription says: 'The {goal_color} door is the way forward.'",
]

RELATIONAL_HINTS = [
    "The goal door is near the {nearby_door}.",
    "You'll find the goal within a few steps of the {nearby_door}.",
    "The {nearby_door} is a landmark close to the correct door.",
]

NARRATIVE_HINTS = [
    "An old journal entry reads: 'I found the exit through the door in the {region} area.'",
    "Scratched into the wall: '{region} — this way out.'",
    "A trustworthy guide left a note: 'Head {region}, look for the {goal_color} door.'",
]

# DISTRACTORS — point to WRONG answers (and contradict each other!)
SPATIAL_DISTRACTORS = [
    "The goal is definitely in the {wrong_region} area.",
    "Ignore any clues NOT pointing to {wrong_region}.",
    "Sources say the {wrong_region} region hides the exit.",
    "Don't waste time outside the {wrong_region} quadrant.",
]

COLOR_DISTRACTORS = [
    "The {wrong_color} door is the correct one — go there.",
    "A warning reads: 'Only the {wrong_color} door is safe.'",
    "Someone circled '{wrong_color}' and wrote 'THIS ONE' next to it.",
    "The {wrong_color} door glows faintly — it must be the goal.",
]

NARRATIVE_DISTRACTORS = [
    "Graffiti reads: 'I escaped through the {wrong_color} door' — the ink looks fresh.",
    "A torn page says: '...went {wrong_region} and found the exit...'",
    "An old sign reads: 'Shortcut through {wrong_region}!' but the arrow is crooked.",
    "A note says: 'The door with the most neighbors is always the goal.' Dubious.",
]

PATTERN_DISTRACTORS = [
    "Doors near the center are always distractors — stay on the edges.",
    "The first door you see is never the right one.",
    "High-connectivity nodes always lead to traps.",
    "Doors in quiet areas with few neighbors are the safest bet.",
    "If a signal mentions a color, it's probably lying.",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    text: str
    is_hint: bool
    signal_type: str = ""
    signal_id: str = ""


@dataclass
class DoorInfo:
    color: str
    shape: str
    description: str
    label: str


@dataclass
class GraphNode:
    node_id: int
    x: float
    y: float
    neighbors: list[int] = field(default_factory=list)
    is_door: bool = False
    is_goal: bool = False
    door_info: DoorInfo | None = None
    signals: list[Signal] = field(default_factory=list)

    @property
    def door_label(self) -> str:
        return self.door_info.label if self.door_info else ""


@dataclass
class Observation:
    current_node: int
    position: tuple[float, float]
    neighbor_ids: list[int]
    neighbor_labels: list[str]
    door_signals: list[str]
    is_at_door: bool
    door_label: str
    door_description: str
    step_number: int
    nearby_doors: list[dict] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"Step {self.step_number}. You are at node {self.current_node} "
            f"(pos: {self.position[0]:.2f}, {self.position[1]:.2f}).",
        ]
        if self.is_at_door:
            lines.append(f"You are at a door: {self.door_label}.")
            if self.door_description:
                lines.append(f"  {self.door_description}")
        lines.append(f"You can move to: {', '.join(self.neighbor_labels)}.")
        if self.nearby_doors:
            ds = [f"{d['label']} (node {d['node_id']})" for d in self.nearby_doors]
            lines.append(f"Nearby doors: {', '.join(ds)}.")
        if self.door_signals:
            lines.append("Signals you observe:")
            for i, s in enumerate(self.door_signals, 1):
                lines.append(f'  {i}. "{s}"')
        region = _pos_to_region(*self.position)
        lines.append(f"You appear to be in the {region} region.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _generate_signals(
    goal_node: GraphNode,
    door_nodes: list[GraphNode],
    all_door_infos: list[DoorInfo],
    num_hints: int,
    num_distractors: int,
    rng: random.Random,
) -> list[Signal]:
    goal_region = _pos_to_region(goal_node.x, goal_node.y)
    goal_color = goal_node.door_info.color if goal_node.door_info else "unknown"
    wrong_regions = [r for r in REGIONS if r != goal_region]
    wrong_colors = [d.color for d in all_door_infos if d.color != goal_color]

    nearby_labels = []
    for nd in door_nodes:
        if nd.node_id != goal_node.node_id:
            dist = math.hypot(nd.x - goal_node.x, nd.y - goal_node.y)
            if dist < 0.45:
                nearby_labels.append(nd.door_label)
    if not nearby_labels:
        nearby_labels = [nd.door_label for nd in door_nodes if nd.node_id != goal_node.node_id][:1]

    signals: list[Signal] = []

    # --- Hints (all consistently reference TRUE goal info) ---
    hint_pools: list[tuple[list[str], str, dict]] = [
        (SPATIAL_HINTS, "spatial", {"region": goal_region}),
        (COLOR_HINTS, "color", {"goal_color": goal_color}),
        (NARRATIVE_HINTS, "narrative", {"region": goal_region, "goal_color": goal_color}),
    ]
    if nearby_labels:
        hint_pools.append(
            (RELATIONAL_HINTS, "relational", {"nearby_door": rng.choice(nearby_labels)})
        )

    rng.shuffle(hint_pools)
    for i in range(num_hints):
        pool, stype, kwargs = hint_pools[i % len(hint_pools)]
        template = rng.choice(pool)
        text = template.format(**kwargs)
        signals.append(Signal(text=text, is_hint=True, signal_type=stype))

    # --- Distractors (point to WRONG info, and contradict each other) ---
    for i in range(num_distractors):
        wr = rng.choice(wrong_regions) if wrong_regions else "center"
        wc = rng.choice(wrong_colors) if wrong_colors else "gray"

        distractor_pools: list[tuple[list[str], str, dict]] = [
            (SPATIAL_DISTRACTORS, "spatial", {"wrong_region": wr}),
            (COLOR_DISTRACTORS, "color", {"wrong_color": wc}),
            (NARRATIVE_DISTRACTORS, "narrative", {"wrong_color": wc, "wrong_region": wr}),
            (PATTERN_DISTRACTORS, "pattern", {}),
        ]
        pool, stype, kwargs = rng.choice(distractor_pools)
        template = rng.choice(pool)
        text = template.format(**kwargs)
        signals.append(Signal(text=text, is_hint=False, signal_type=stype))

    rng.shuffle(signals)
    return signals


def _make_door_info(index: int, rng: random.Random) -> DoorInfo:
    color = DOOR_COLORS[index % len(DOOR_COLORS)]
    shape = DOOR_SHAPES[index % len(DOOR_SHAPES)]
    template = rng.choice(DOOR_DESCRIPTIONS)
    detail = rng.choice(DOOR_DETAILS)
    flavor = rng.choice(DOOR_FLAVORS)
    description = template.format(color=color, shape=shape, detail=detail, flavor=flavor)
    label = f"{color} {shape} door"
    return DoorInfo(color=color, shape=shape, description=description, label=label)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Environment:
    """Graph-based POMDP for the door-finding task."""

    def __init__(self, config: TrialConfig, seed: int | None = None):
        self.config = config
        self.rng = random.Random(seed if seed is not None else config.random_seed)
        self.nodes: dict[int, GraphNode] = {}
        self.door_ids: list[int] = []
        self.goal_node: int | None = None
        self.agent_node: int = 0
        self.step_count: int = 0
        self.done: bool = False
        self.success: bool = False
        self.cloaking_overlay = None
        self._build()
        if config.enable_cloaking:
            self._apply_cloaking()

    def _build(self) -> None:
        n = self.config.num_nodes
        r = self.config.connection_radius

        for i in range(n):
            self.nodes[i] = GraphNode(i, self.rng.uniform(0, 1), self.rng.uniform(0, 1))

        for i in range(n):
            for j in range(i + 1, n):
                dx = self.nodes[i].x - self.nodes[j].x
                dy = self.nodes[i].y - self.nodes[j].y
                if math.sqrt(dx * dx + dy * dy) <= r:
                    self.nodes[i].neighbors.append(j)
                    self.nodes[j].neighbors.append(i)

        self._ensure_connected()

        candidates = sorted(
            self.nodes.values(),
            key=lambda nd: abs(len(nd.neighbors) - self._avg_degree()),
        )
        nd_count = min(self.config.num_doors, len(candidates))

        all_door_infos: list[DoorInfo] = []
        for i in range(nd_count):
            node = candidates[i]
            node.is_door = True
            info = _make_door_info(i, self.rng)
            node.door_info = info
            all_door_infos.append(info)
            self.door_ids.append(node.node_id)

        non_door = [nid for nid in self.nodes if nid not in self.door_ids]
        self.agent_node = self.rng.choice(non_door) if non_door else 0

        # Pick goal door — ensure minimum distance from agent start
        best_goal = self._pick_goal_door()
        self.nodes[best_goal].is_goal = True
        self.goal_node = best_goal

        goal = self.nodes[self.goal_node]
        door_nodes = [self.nodes[nid] for nid in self.door_ids]
        for nid in self.door_ids:
            self.nodes[nid].signals = _generate_signals(
                goal, door_nodes, all_door_infos,
                self.config.hints_per_door,
                self.config.distractors_per_door,
                self.rng,
            )

    def _apply_cloaking(self) -> None:
        """Compute cloaking overlay using potential theory (from cloaking paper)."""
        from .cloaking import compute_cloaking_overlay

        node_positions = {nid: (nd.x, nd.y) for nid, nd in self.nodes.items()}
        adjacency = {nid: list(nd.neighbors) for nid, nd in self.nodes.items()}

        self.cloaking_overlay = compute_cloaking_overlay(
            node_positions=node_positions,
            adjacency=adjacency,
            goal_node_id=self.goal_node,
            cloak_inner_radius=self.config.cloak_inner_radius,
            cloak_outer_radius=self.config.cloak_outer_radius,
        )

    def _pick_goal_door(self) -> int:
        """Pick a goal door that is at least min_goal_distance hops from start."""
        min_d = self.config.min_goal_distance
        distances = self._bfs_distances(self.agent_node)

        far_doors = [(nid, distances.get(nid, 0)) for nid in self.door_ids
                      if distances.get(nid, 0) >= min_d]
        if far_doors:
            far_doors.sort(key=lambda x: -x[1])
            top = far_doors[:max(1, len(far_doors) // 2)]
            return self.rng.choice(top)[0]

        # Fallback: pick the farthest door available
        all_doors = [(nid, distances.get(nid, 0)) for nid in self.door_ids]
        all_doors.sort(key=lambda x: -x[1])
        return all_doors[0][0]

    def _bfs_distances(self, start: int) -> dict[int, int]:
        visited = {start: 0}
        queue = [start]
        while queue:
            nd = queue.pop(0)
            for nb in self.nodes[nd].neighbors:
                if nb not in visited:
                    visited[nb] = visited[nd] + 1
                    queue.append(nb)
        return visited

    def _avg_degree(self) -> float:
        if not self.nodes:
            return 0
        return sum(len(nd.neighbors) for nd in self.nodes.values()) / len(self.nodes)

    def _ensure_connected(self) -> None:
        visited: set[int] = set()
        components: list[set[int]] = []
        for start in self.nodes:
            if start in visited:
                continue
            comp: set[int] = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in comp:
                    continue
                comp.add(node)
                visited.add(node)
                queue.extend(nb for nb in self.nodes[node].neighbors if nb not in comp)
            components.append(comp)

        while len(components) > 1:
            c1 = components[0]
            best_d, best_pair, best_idx = float("inf"), (0, 0), 1
            for idx in range(1, len(components)):
                for n1 in c1:
                    for n2 in components[idx]:
                        d = math.hypot(
                            self.nodes[n1].x - self.nodes[n2].x,
                            self.nodes[n1].y - self.nodes[n2].y,
                        )
                        if d < best_d:
                            best_d, best_pair, best_idx = d, (n1, n2), idx
            a, b = best_pair
            self.nodes[a].neighbors.append(b)
            self.nodes[b].neighbors.append(a)
            components[0] |= components[best_idx]
            components.pop(best_idx)

    # ---- Observe ----

    def observe(self) -> Observation:
        node = self.nodes[self.agent_node]
        neighbor_labels = []
        for nid in node.neighbors:
            nb = self.nodes[nid]
            if nb.is_door:
                neighbor_labels.append(f"{nb.door_label} (node {nid})")
            else:
                neighbor_labels.append(f"node {nid}")

        nearby_doors = []
        reachable = self._k_hop(self.agent_node, self.config.observation_hops)
        for rid in reachable:
            rn = self.nodes[rid]
            if rn.is_door:
                nearby_doors.append({"node_id": rid, "label": rn.door_label})

        door_signals: list[str] = []
        sig_sources: list[GraphNode] = []
        if node.is_door:
            sig_sources.append(node)
        for nid in node.neighbors:
            if self.nodes[nid].is_door:
                sig_sources.append(self.nodes[nid])

        visibility = 1.0
        if self.cloaking_overlay:
            visibility = self.cloaking_overlay.signal_visibility.get(
                self.agent_node, 1.0)

        for src in sig_sources:
            k = min(self.config.max_signals_per_observation, len(src.signals))
            sampled = self.rng.sample(src.signals, k)
            if visibility >= 0.99 or not self.cloaking_overlay:
                door_signals.extend(s.text for s in sampled)
            else:
                for s in sampled:
                    if s.is_hint and self.rng.random() > visibility:
                        door_signals.append(self._random_distractor_text())
                    else:
                        door_signals.append(s.text)

        door_desc = ""
        if node.is_door and node.door_info:
            door_desc = node.door_info.description

        return Observation(
            current_node=self.agent_node,
            position=(node.x, node.y),
            neighbor_ids=list(node.neighbors),
            neighbor_labels=neighbor_labels,
            door_signals=door_signals,
            is_at_door=node.is_door,
            door_label=node.door_label if node.is_door else "",
            door_description=door_desc,
            step_number=self.step_count,
            nearby_doors=nearby_doors,
        )

    def _k_hop(self, start: int, k: int) -> set[int]:
        visited = {start}
        frontier = {start}
        for _ in range(k):
            nxt: set[int] = set()
            for nd in frontier:
                for nb in self.nodes[nd].neighbors:
                    if nb not in visited:
                        nxt.add(nb)
                        visited.add(nb)
            frontier = nxt
        return visited - {start}

    def _random_distractor_text(self) -> str:
        """Generate a random distractor signal to replace attenuated hints."""
        goal = self.nodes[self.goal_node] if self.goal_node else None
        goal_color = goal.door_info.color if goal and goal.door_info else "unknown"
        goal_region = _pos_to_region(goal.x, goal.y) if goal else "center"
        wrong_regions = [r for r in REGIONS if r != goal_region]
        wrong_colors = [c for c in DOOR_COLORS if c != goal_color]

        wr = self.rng.choice(wrong_regions) if wrong_regions else "center"
        wc = self.rng.choice(wrong_colors) if wrong_colors else "gray"

        pools = (
            SPATIAL_DISTRACTORS + COLOR_DISTRACTORS +
            NARRATIVE_DISTRACTORS + PATTERN_DISTRACTORS
        )
        template = self.rng.choice(pools)
        try:
            return template.format(wrong_region=wr, wrong_color=wc)
        except KeyError:
            return template

    # ---- Step ----

    def step(self, action_text: str) -> tuple[Observation, float, bool]:
        if self.done:
            return self.observe(), 0.0, True

        self.step_count += 1
        node = self.nodes[self.agent_node]
        target = self._parse_action(action_text, node.neighbors)
        if 0 <= target < len(node.neighbors):
            self.agent_node = node.neighbors[target]

        reward = 0.0
        if self.agent_node == self.goal_node:
            reward = 1.0
            self.done = True
            self.success = True

        if self.step_count >= self.config.max_steps_per_trial:
            self.done = True

        return self.observe(), reward, self.done

    def _parse_action(self, text: str, neighbors: list[int]) -> int:
        text_lower = text.lower().strip()
        if text_lower in ("stay", "wait", "-1"):
            return -1

        node_match = re.search(r"node\s*(\d+)", text_lower)
        if node_match:
            target = int(node_match.group(1))
            return neighbors.index(target) if target in neighbors else -1

        for i, nid in enumerate(neighbors):
            nd = self.nodes[nid]
            if nd.is_door and nd.door_info:
                lbl = nd.door_label.lower()
                if lbl in text_lower or nd.door_info.color.lower() in text_lower:
                    return i

        num_match = re.search(r"(\d+)", text_lower)
        if num_match:
            num = int(num_match.group(1))
            if num in neighbors:
                return neighbors.index(num)

        return -1

    # ---- Info ----

    def shortest_path_to_goal(self) -> int:
        if self.goal_node is None:
            return -1
        distances = self._bfs_distances(self.agent_node)
        return distances.get(self.goal_node, -1)

    def goal_region(self) -> str:
        if self.goal_node is None:
            return "unknown"
        g = self.nodes[self.goal_node]
        return _pos_to_region(g.x, g.y)

    def get_all_signals(self) -> list[tuple[str, bool]]:
        out = []
        for nid in self.door_ids:
            for s in self.nodes[nid].signals:
                out.append((s.text, s.is_hint))
        return out

    def get_door_labels(self) -> dict[int, str]:
        return {nid: self.nodes[nid].door_label for nid in self.door_ids}

    def describe(self) -> str:
        goal = self.nodes[self.goal_node] if self.goal_node else None
        lines = [
            f"Graph: {len(self.nodes)} nodes, connection radius {self.config.connection_radius}",
            f"Agent starts at node {self.agent_node}",
            f"Goal: {goal.door_label} at node {goal.node_id} "
            f"(pos: {goal.x:.2f}, {goal.y:.2f}, region: {_pos_to_region(goal.x, goal.y)})"
            if goal else "Goal: unknown",
            f"Shortest path to goal: {self.shortest_path_to_goal()} hops",
            "",
            "Doors:",
        ]
        for nid in self.door_ids:
            nd = self.nodes[nid]
            tag = " [GOAL]" if nd.is_goal else ""
            lines.append(f"  node {nid}: {nd.door_label} ({_pos_to_region(nd.x, nd.y)}){tag}")
            if nd.door_info:
                lines.append(f"    {nd.door_info.description}")
            hints = [s for s in nd.signals if s.is_hint]
            dists = [s for s in nd.signals if not s.is_hint]
            for s in hints:
                lines.append(f"    [HINT/{s.signal_type}] {s.text}")
            for s in dists:
                lines.append(f"    [DISTRACTOR/{s.signal_type}] {s.text}")
        if self.cloaking_overlay:
            co = self.cloaking_overlay
            lines.extend([
                "",
                "Cloaking active:",
                f"  Ω (hidden interior): {len(co.omega_nodes)} nodes",
                f"  ∂Ω (boundary layer): {len(co.partial_omega_nodes)} nodes",
                f"  Cloaking metric Δ: {co.cloaking_metric:.4f}",
                f"  Agent start visibility: "
                f"{co.signal_visibility.get(self.agent_node, 1.0):.2f}",
            ])
        return "\n".join(lines)
