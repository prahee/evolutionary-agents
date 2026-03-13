"""Trial runner: the core simulation loop.

Key fixes over the previous version:
  1. Agents reproduce ON SUCCESS (before being retired) so knowledge transfers
  2. Multiple reproduction triggers (periodic, on success)
  3. Optional Bayesian belief tracking
  4. Tracks reproduction trigger types for analysis
  5. 2 root agents by default for population diversity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_dartmouth.llms import ChatDartmouth

from .agent import Agent
from .beliefs import BeliefState, extract_evidence
from .config import TrialConfig
from .environment import Environment
from .logger import TranscriptLogger
from .reproduction import BirthEvent, Lineage, birth_agent, next_id, reset_id_counter
from .skill_library import SkillLibrary


@dataclass
class AgentState:
    agent: Agent
    env: Environment
    interactions: int = 0
    succeeded: bool = False


@dataclass
class TrialResult:
    config: dict[str, Any] = field(default_factory=dict)
    total_steps: int = 0
    successful_agents: list[str] = field(default_factory=list)
    stopped_reason: str = "success"
    lineage: Lineage = field(default_factory=Lineage)
    priors: dict[str, str] = field(default_factory=dict)
    skill_library_snapshot: list[dict] = field(default_factory=list)
    per_agent_data: dict[str, dict] = field(default_factory=dict)
    all_signals: list[tuple[str, bool]] = field(default_factory=list)
    transcript_path: str | None = None
    belief_trajectories: dict[str, list[dict]] = field(default_factory=dict)


class TrialRunner:

    def __init__(self, config: TrialConfig, transcript_dir: Path | None = None,
                 initial_priors: list[str] | None = None):
        self.config = config
        self.initial_priors = initial_priors or []
        self.reasoning_llm = ChatDartmouth(model_name=config.reasoning_model)
        self.utility_llm = ChatDartmouth(model_name=config.utility_model)

        self.active: list[AgentState] = []
        self.successful: list[str] = []
        self.lineage = Lineage()
        self.skill_library = SkillLibrary() if config.enable_skill_library else None
        self.total_steps = 0
        self.priors: dict[str, str] = {}
        self.parent_refs: dict[str, Agent] = {}
        self.all_signals: list[tuple[str, bool]] = []
        self.belief_trajectories: dict[str, list[dict]] = {}

        self.logger = TranscriptLogger() if config.log_transcript else None
        self.transcript_dir = transcript_dir
        reset_id_counter()

    def _make_env(self, seed_override: int | None = None) -> Environment:
        return Environment(self.config, seed=seed_override or self.config.random_seed)

    def _make_agent(self, agent_id: str, prior: str, parent_id: str | None,
                    generation: int) -> Agent:
        return Agent(
            agent_id=agent_id,
            config=self.config,
            prior=prior,
            parent_id=parent_id,
            generation=generation,
            reasoning_llm=self.reasoning_llm,
            utility_llm=self.utility_llm,
            logger=self.logger,
        )

    def _make_root_agent(self, prior: str = "") -> Agent:
        return self._make_agent(next_id(), prior, None, 0)

    def _init_belief_state(self, agent: Agent, env: Environment) -> None:
        if not self.config.enable_bayesian:
            return
        door_labels = env.get_door_labels()
        agent.belief_state = BeliefState(
            door_ids=list(door_labels.keys()),
            door_labels=door_labels,
        )

    def _update_beliefs(self, agent: Agent, obs) -> None:
        if not self.config.enable_bayesian or agent.belief_state is None:
            return
        if not obs.door_signals:
            return
        evidence = extract_evidence(
            self.utility_llm, obs.to_text(), agent.belief_state.door_labels,
        )
        for door_id, reliability in evidence:
            agent.belief_state.update_from_evidence(door_id, reliability)

    def _do_birth(self, parent: Agent, trigger: str, env: Environment) -> Agent | None:
        """Birth a child from parent. Returns the child or None if limit reached."""
        if not parent.can_have_more_children():
            return None

        prior = parent.compress_to_prior()
        ctx_snapshot = parent.get_full_context()
        self.priors[parent.agent_id] = prior

        child = birth_agent(parent, prior, self.config,
                            self.reasoning_llm, self.utility_llm)
        child.logger = self.logger

        self.lineage.add_birth(BirthEvent(
            parent_id=parent.agent_id,
            child_id=child.agent_id,
            generation=child.generation,
            step=self.total_steps,
            trigger=trigger,
            parent_prior=parent.prior,
            child_prior=prior if self.config.inherit_prior else "",
            parent_context_snapshot=ctx_snapshot,
        ))
        self.parent_refs[parent.agent_id] = parent

        if self.logger:
            self.logger.log_birth(
                parent_id=parent.agent_id,
                child_id=child.agent_id,
                parent_generation=parent.generation,
                child_generation=child.generation,
                step=self.total_steps,
                child_prior=child.prior,
                trigger=trigger,
            )

        if self.skill_library:
            conv = parent.propose_convention()
            if conv:
                accepted = self.skill_library.add(
                    conv, parent.agent_id, parent.generation, self.total_steps,
                )
                if self.logger:
                    self.logger.log_skill_library_add(parent.agent_id, conv, accepted)

        return child

    def run(self, on_step: callable | None = None) -> TrialResult:
        # Initialize root agents (optionally with injected priors)
        for i in range(self.config.num_root_agents):
            injected = self.initial_priors[i] if i < len(self.initial_priors) else ""
            agent = self._make_root_agent(prior=injected)
            env = self._make_env()
            self._init_belief_state(agent, env)
            self.lineage.add_root(agent.agent_id)
            self.active.append(AgentState(agent=agent, env=env))
            if not self.all_signals:
                self.all_signals = env.get_all_signals()
                if self.logger:
                    self.logger.log_env_setup(env.describe())

        # Main loop
        while True:
            if len(self.successful) >= self.config.success_count:
                break
            if self.total_steps >= self.config.max_steps:
                break
            if not self.active:
                break

            new_states: list[AgentState] = []

            for state in self.active:
                if state.succeeded:
                    continue
                if self.total_steps >= self.config.max_steps:
                    break

                obs = state.env.observe()

                # Bayesian belief update
                self._update_beliefs(state.agent, obs)

                # Parent querying
                if (self.config.enable_parent_query
                        and state.agent.parent_id
                        and state.agent.parent_queries_used < self.config.max_parent_queries
                        and state.agent.should_query_parent()):
                    parent = self.parent_refs.get(state.agent.parent_id)
                    if parent:
                        question = state.agent.formulate_parent_question(obs)
                        state.agent.ask_parent(parent, question)

                # Skill library query
                skill_ctx = ""
                if self.skill_library:
                    skill_ctx = self.skill_library.query(obs.to_text())

                action_text = state.agent.decide(obs, skill_context=skill_ctx,
                                                  global_step=self.total_steps)
                _, reward, done = state.env.step(action_text)
                state.interactions += 1
                self.total_steps += 1

                # Record belief trajectory
                if state.agent.belief_state:
                    goal_node = state.env.goal_node
                    self.belief_trajectories.setdefault(state.agent.agent_id, []).append({
                        "step": self.total_steps,
                        "entropy": state.agent.belief_state.entropy(),
                        "max_belief": state.agent.belief_state.max_belief(),
                        "goal_belief": state.agent.belief_state.belief_on_true_goal(goal_node),
                        "map_correct": state.agent.belief_state.map_door() == goal_node,
                    })

                # --- SUCCESS ---
                if reward > 0:
                    state.succeeded = True
                    self.successful.append(state.agent.agent_id)

                    prior = state.agent.compress_to_prior()
                    self.priors[state.agent.agent_id] = prior

                    if self.logger:
                        self.logger.log_success(
                            state.agent.agent_id, state.agent.generation,
                            self.total_steps,
                        )

                    if self.skill_library:
                        conv = state.agent.propose_convention()
                        if conv:
                            accepted = self.skill_library.add(
                                conv, state.agent.agent_id,
                                state.agent.generation, self.total_steps,
                            )
                            if self.logger:
                                self.logger.log_skill_library_add(
                                    state.agent.agent_id, conv, accepted,
                                )

                    # CRITICAL: reproduce on success so knowledge transfers!
                    if self.config.reproduce_on_success:
                        child = self._do_birth(state.agent, "on_success", state.env)
                        if child:
                            child_env = self._make_env()
                            self._init_belief_state(child, child_env)
                            new_states.append(AgentState(agent=child, env=child_env))

                    continue  # parent retires (succeeded)

                # --- PERIODIC REPRODUCTION ---
                if state.agent.should_reproduce_periodic():
                    child = self._do_birth(state.agent, "periodic", state.env)
                    if child:
                        child_env = self._make_env()
                        self._init_belief_state(child, child_env)
                        new_states.append(AgentState(agent=child, env=child_env))
                    state.interactions = 0
                    state.agent.interactions = 0

                # --- NOVELTY-TRIGGERED REPRODUCTION ---
                elif state.agent.should_reproduce_novelty():
                    child = self._do_birth(state.agent, "novelty", state.env)
                    if child:
                        child_env = self._make_env()
                        self._init_belief_state(child, child_env)
                        new_states.append(AgentState(agent=child, env=child_env))
                    state.agent.interactions = 0

                new_states.append(state)

            self.active = [s for s in new_states if not s.succeeded]

            if on_step:
                on_step(self.total_steps, len(self.successful), len(self.active))

        # Log final state
        if self.logger:
            for state in self.active:
                self.logger.log_final_context(
                    state.agent.agent_id, state.agent.get_full_context(),
                )
            for aid, agent in self.parent_refs.items():
                self.logger.log_final_context(aid, agent.get_full_context())
            self.logger.log_lineage_tree(
                self.lineage.tree_str(set(self.successful))
            )
            if self.skill_library:
                self.logger.section("SKILL LIBRARY (final)")
                self.logger._write(self.skill_library.summary())

        # Save transcript
        transcript_path = None
        if self.logger and self.transcript_dir:
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
            seed_tag = self.config.random_seed or "noseed"
            tp = self.transcript_dir / f"transcript_seed{seed_tag}.txt"
            self.logger.save(tp)
            transcript_path = str(tp)

        # Collect per-agent data
        per_agent: dict[str, dict] = {}
        seen_ids: set[str] = set()
        all_agents = [s.agent for s in self.active] + list(self.parent_refs.values())
        for agent in all_agents:
            if agent.agent_id in seen_ids:
                continue
            seen_ids.add(agent.agent_id)
            per_agent[agent.agent_id] = {
                "generation": agent.generation,
                "interactions": agent.interactions,
                "prior": agent.prior,
                "actions": agent.actions_taken,
                "parent_id": agent.parent_id,
                "parent_queries_used": agent.parent_queries_used,
                "children_birthed": agent.children_birthed,
                "doors_visited": list(agent.doors_visited),
                "context_overflows": agent.context_overflows,
                "context_snapshot": agent.get_context_snapshot(),
            }

        stopped = ("success" if len(self.successful) >= self.config.success_count
                    else "max_steps")

        return TrialResult(
            config={
                "num_nodes": self.config.num_nodes,
                "num_doors": self.config.num_doors,
                "hints_per_door": self.config.hints_per_door,
                "distractors_per_door": self.config.distractors_per_door,
                "inherit_prior": self.config.inherit_prior,
                "enable_parent_query": self.config.enable_parent_query,
                "enable_skill_library": self.config.enable_skill_library,
                "enable_bayesian": self.config.enable_bayesian,
                "reproduce_on_success": self.config.reproduce_on_success,
                "max_prior_tokens": self.config.max_prior_tokens,
                "max_context_tokens": self.config.max_context_tokens,
                "interactions_per_lifetime": self.config.interactions_per_lifetime,
                "max_steps": self.config.max_steps,
                "num_root_agents": self.config.num_root_agents,
                "min_goal_distance": self.config.min_goal_distance,
                "enable_cloaking": self.config.enable_cloaking,
                "cloak_inner_radius": self.config.cloak_inner_radius,
                "cloak_outer_radius": self.config.cloak_outer_radius,
                "reasoning_model": self.config.reasoning_model,
                "utility_model": self.config.utility_model,
            },
            total_steps=self.total_steps,
            successful_agents=self.successful,
            stopped_reason=stopped,
            lineage=self.lineage,
            priors=self.priors,
            skill_library_snapshot=(self.skill_library.to_dicts()
                                    if self.skill_library else []),
            per_agent_data=per_agent,
            all_signals=self.all_signals,
            transcript_path=transcript_path,
            belief_trajectories=self.belief_trajectories,
        )
