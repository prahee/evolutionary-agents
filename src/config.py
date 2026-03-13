"""Experiment configuration with multiple reproduction triggers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def get_allowed_models() -> list[str]:
    path = Path(__file__).resolve().parent.parent / "models.txt"
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def validate_model(name: str) -> None:
    allowed = get_allowed_models()
    if allowed and name not in allowed:
        raise ValueError(f"Model '{name}' not in models.txt")


@dataclass
class TrialConfig:
    """Configuration for a single experimental trial."""

    # --- Environment (graph POMDP) ---
    num_nodes: int = 20
    connection_radius: float = 0.35
    num_doors: int = 5
    hints_per_door: int = 3
    distractors_per_door: int = 2
    max_signals_per_observation: int = 4
    observation_hops: int = 1
    random_seed: int | None = None
    min_goal_distance: int = 3

    # --- Agent context ---
    max_context_tokens: int = 750
    max_prior_tokens: int = 150

    # --- Reproduction ---
    interactions_per_lifetime: int = 10
    reproduce_on_success: bool = True
    reproduce_on_novelty: bool = False
    novelty_threshold: float = 0.7
    max_children_per_agent: int = 5

    # --- Features ---
    inherit_prior: bool = True
    enable_parent_query: bool = False
    max_parent_queries: int = 3
    parent_query_steps: tuple[int, ...] = (0, 3, 7)
    enable_skill_library: bool = False
    enable_bayesian: bool = False

    # --- Cloaking (potential-theory signal attenuation) ---
    enable_cloaking: bool = False
    cloak_inner_radius: float = 0.25
    cloak_outer_radius: float = 0.40

    # --- Lifecycle ---
    max_steps: int = 500
    success_count: int = 3
    num_root_agents: int = 2

    # --- LLM ---
    reasoning_model: str = "openai.gpt-4.1-mini-2025-04-14"
    utility_model: str = "vertex_ai.gemini-2.0-flash-001"
    max_steps_per_trial: int = 300

    # --- Logging ---
    log_transcript: bool = True


def validate_config_models(config: TrialConfig) -> None:
    validate_model(config.reasoning_model)
    validate_model(config.utility_model)


@dataclass
class ExperimentConfig:
    """Configuration for a multi-trial experiment."""

    name: str = "experiment"
    num_trials: int = 5
    trial: TrialConfig = field(default_factory=TrialConfig)
    output_dir: str = "results"

    sweep_param: str | None = None
    sweep_values: list = field(default_factory=list)
    conditions: dict[str, dict] = field(default_factory=dict)
