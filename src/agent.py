"""LLM agent with context management, prior inheritance, optional Bayesian
belief tracking, parent querying, and full transcript logging.

Uses two model tiers:
  - reasoning_llm: decisions, prior compression, parent Q&A, conventions
  - utility_llm:   context summarisation, question formulation, evidence extraction
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_dartmouth.llms import ChatDartmouth

from .config import TrialConfig
from .environment import Observation

if TYPE_CHECKING:
    from .beliefs import BeliefState
    from .logger import TranscriptLogger

# ---------------------------------------------------------------------------
# Robust LLM invocation with retry
# ---------------------------------------------------------------------------

def invoke_with_retry(llm, messages, max_retries: int = 3, backoff: float = 2.0) -> str:
    """Call llm.invoke() with retry on transient failures (None responses, API errors)."""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            if text is None:
                raise ValueError("LLM returned None content")
            return text.strip()
        except Exception:
            if attempt == max_retries - 1:
                return "(LLM call failed after retries)"
            time.sleep(backoff * (attempt + 1))
    return "(LLM call failed)"


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an agent navigating a graph to find the correct goal door.
You see your position, neighboring nodes, and signals at nearby doors.
Doors have colors, shapes, and descriptions. Signals may be true hints
or misleading distractors — learn which to trust by looking for consistency
(true hints agree with each other, distractors contradict each other).

Choose your next move by saying the node ID or door label to move to.
Example: "move to node 7" or "move to red arched door" or "stay".
After your action, add a brief reasoning line starting with "Reason:" (1-2 sentences).

IMPORTANT: If you are AT a door and believe it is the goal, say "enter <door label>"
to attempt it. You must enter a door to succeed — just standing at it is not enough."""

SUMMARIZE_PROMPT = """\
Compress these exploration notes into 2-3 concise sentences.
Prioritize:
- General strategies: what patterns make a signal trustworthy vs misleading
- Lessons about signal reliability (e.g., consistent signals agree, distractors contradict)
- Navigation heuristics that transfer to similar environments
- Shorthand or naming conventions you've developed
Drop: exact coordinates, specific node IDs, redundant steps, verbose descriptions.
Retain any emergent shorthand or abbreviations you've been using."""

COMPRESS_PRIOR_PROMPT = """\
Write a concise prior (3-5 sentences, under {token_limit} words) for a new agent.
Include: (1) which signals/colors to trust vs ignore, (2) navigation strategy,
(3) key environmental patterns you learned, (4) which doors to avoid.
Be specific and actionable. This is the child agent's only inherited knowledge."""

COMPRESS_PRIOR_ABSTRACT_PROMPT = """\
Write a concise survival guide (3-5 sentences, under {token_limit} words) for a new
agent who will face a DIFFERENT environment from yours.
Focus on GENERAL, TRANSFERABLE strategies:
(1) How to distinguish reliable hints from distractors (e.g., real hints agree
    with each other; distractors contradict each other).
(2) General navigation heuristics (explore broadly first, then converge).
(3) What TYPES of signals tend to be trustworthy vs misleading.
(4) Meta-strategies for learning in unknown environments.
Do NOT mention specific door colors, node IDs, or region names.
This guide must help in ANY door-finding environment, not just yours."""

PARENT_QUERY_PROMPT = """\
A child agent is asking you a question based on your exploration experience.
Answer concisely (1-2 sentences) based on what you've learned.
Child's question: {question}"""

CONVENTION_PROMPT = """\
Based on your exploration, state ONE general rule about this environment that
would help other agents. Think about:
- Which door colors or types tend to be goals vs distractors
- Which signal patterns are reliable vs misleading
- Spatial patterns (e.g. "goal is usually far from the starting area")
- What signals consistently agree vs contradict each other

State it as a single clear sentence. If nothing useful, say 'none'.

Your experience:
{experience}"""


@dataclass
class ContextEntry:
    text: str
    is_summary: bool = False
    token_est: int = 0


@dataclass
class Agent:
    agent_id: str
    config: TrialConfig
    prior: str = ""
    parent_id: str | None = None
    generation: int = 0

    reasoning_llm: ChatDartmouth | None = None
    utility_llm: ChatDartmouth | None = None
    logger: TranscriptLogger | None = None
    belief_state: BeliefState | None = None

    context: list[ContextEntry] = field(default_factory=list)
    total_tokens: int = 0
    interactions: int = 0
    parent_queries_used: int = 0
    children_birthed: int = 0
    context_overflows: int = 0

    actions_taken: list[str] = field(default_factory=list)
    doors_visited: set = field(default_factory=set)

    def __post_init__(self):
        if self.reasoning_llm is None:
            self.reasoning_llm = ChatDartmouth(model_name=self.config.reasoning_model)
        if self.utility_llm is None:
            self.utility_llm = ChatDartmouth(model_name=self.config.utility_model)

    # ---- Decision (reasoning model) ----

    def decide(self, obs: Observation, skill_context: str = "", global_step: int = 0) -> str:
        if obs.is_at_door:
            self.doors_visited.add(obs.door_label)

        messages = self._build_messages(obs, skill_context)
        raw = invoke_with_retry(self.reasoning_llm, messages)

        action_line, reasoning = self._parse_response(raw)

        entry_text = f"Obs: {obs.to_text()}\nAction: {action_line}\nReason: {reasoning}"
        self._add_context(entry_text)
        self.interactions += 1
        self.actions_taken.append(action_line)

        if self.logger:
            self.logger.log_decision(
                agent_id=self.agent_id,
                generation=self.generation,
                step=global_step,
                observation_text=obs.to_text(),
                skill_library_context=skill_context,
                llm_response=raw,
                parsed_action=action_line,
                reasoning=reasoning,
            )

        return action_line

    def _build_messages(self, obs: Observation, skill_context: str = "") -> list:
        sys = SYSTEM_PROMPT
        if self.prior:
            sys += f"\n\nInherited knowledge from your parent:\n{self.prior}"
        if self.belief_state:
            sys += f"\n\nCurrent belief distribution over doors:\n{self.belief_state.to_text()}"
        if skill_context:
            sys += f"\n\nRelevant conventions from the skill library:\n{skill_context}"
        messages = [SystemMessage(content=sys)]
        for entry in self.context[-6:]:
            messages.append(HumanMessage(content=entry.text))
        messages.append(HumanMessage(
            content=f"Current observation:\n{obs.to_text()}\n\nChoose your next move:"
        ))
        return messages

    def _parse_response(self, content: str) -> tuple[str, str]:
        lines = content.strip().split("\n")
        action_line = lines[0].strip() if lines else "stay"
        reasoning = ""
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.lower().startswith("reason:"):
                reasoning = stripped[7:].strip()
                break
            elif stripped:
                reasoning = stripped
                break
        return action_line, reasoning or "No reasoning."

    # ---- Context management (utility model) ----

    def _add_context(self, text: str) -> None:
        est = _est_tokens(text)
        self.context.append(ContextEntry(text=text, token_est=est))
        self.total_tokens += est
        while self.total_tokens > self.config.max_context_tokens * 0.8 and len(self.context) > 3:
            self._summarize_oldest()
            self.context_overflows += 1

    def _summarize_oldest(self) -> None:
        half = max(2, len(self.context) // 2)
        to_compress = self.context[:half]
        combined = "\n---\n".join(e.text for e in to_compress)
        prompt = f"{SUMMARIZE_PROMPT}\n\n{combined}"
        summary = invoke_with_retry(self.utility_llm, [HumanMessage(content=prompt)])[:400]
        est = _est_tokens(summary)
        removed = sum(e.token_est for e in to_compress)
        self.context = [
            ContextEntry(text=f"[Summary] {summary}", is_summary=True, token_est=est)
        ] + self.context[half:]
        self.total_tokens = self.total_tokens - removed + est

        if self.logger:
            self.logger.log_summarization(
                agent_id=self.agent_id,
                num_entries_compressed=half,
                original_text_preview=combined,
                summary=summary,
            )

    # ---- Prior compression (reasoning model) ----

    def compress_to_prior(self, abstract: bool = False) -> str:
        if not self.context:
            return self.prior or "No experience yet."
        recent = "\n---\n".join(e.text for e in self.context[-12:])
        token_limit = self.config.max_prior_tokens
        template = COMPRESS_PRIOR_ABSTRACT_PROMPT if abstract else COMPRESS_PRIOR_PROMPT
        prompt = template.format(token_limit=token_limit) + f"\n\nExperience:\n{recent}"

        if self.belief_state:
            prompt += f"\n\nYour current belief state:\n{self.belief_state.to_text()}"
            prompt += f"\n{self.belief_state.to_prior_text()}"

        prior = invoke_with_retry(self.reasoning_llm, [HumanMessage(content=prompt)])
        words = prior.split()
        if len(words) > token_limit:
            prior = " ".join(words[:token_limit])

        if self.logger:
            self.logger.log_prior_compression(
                agent_id=self.agent_id,
                generation=self.generation,
                context_snapshot=self.get_context_snapshot(),
                compressed_prior=prior,
            )

        return prior

    def get_context_snapshot(self) -> str:
        if not self.context:
            return "(empty context)"
        lines = []
        for i, e in enumerate(self.context):
            tag = "[SUM]" if e.is_summary else f"[{i+1}]"
            lines.append(f"{tag} {e.text[:200]}{'...' if len(e.text) > 200 else ''}")
        return "\n".join(lines)

    def get_full_context(self) -> str:
        if not self.context:
            return "(empty context)"
        lines = []
        for i, e in enumerate(self.context):
            tag = "[SUM]" if e.is_summary else f"[{i+1}]"
            lines.append(f"{tag} {e.text}")
        return "\n".join(lines)

    # ---- Parent querying (reasoning model) ----

    def ask_parent(self, parent: Agent, question: str) -> str:
        if self.parent_queries_used >= self.config.max_parent_queries:
            return "(query budget exhausted)"
        self.parent_queries_used += 1

        parent_sys = SYSTEM_PROMPT
        if parent.prior:
            parent_sys += f"\n\nYour accumulated knowledge:\n{parent.prior}"
        messages = [SystemMessage(content=parent_sys)]
        for entry in parent.context[-8:]:
            messages.append(HumanMessage(content=entry.text))
        messages.append(HumanMessage(content=PARENT_QUERY_PROMPT.format(question=question)))

        answer = invoke_with_retry(parent.reasoning_llm, messages)[:200]

        self._add_context(f"Asked parent: {question}\nParent said: {answer}")

        if self.logger:
            self.logger.log_parent_query(
                child_id=self.agent_id,
                parent_id=parent.agent_id,
                child_step=self.interactions,
                question=question,
                answer=answer,
            )

        return answer

    def formulate_parent_question(self, obs: Observation) -> str:
        context_summary = ""
        if self.context:
            context_summary = "\n\nYour experience so far:\n" + "\n".join(
                e.text[:100] for e in self.context[-3:]
            )
        prompt = (
            f"You are exploring a graph and have this observation:\n{obs.to_text()}"
            f"{context_summary}\n\n"
            "What is the single most useful question you could ask a more experienced agent? "
            "Keep it under 25 words."
        )
        return invoke_with_retry(self.utility_llm, [HumanMessage(content=prompt)])[:120]

    def should_query_parent(self) -> bool:
        return self.interactions in self.config.parent_query_steps

    # ---- Skill library (reasoning model) ----

    def propose_convention(self) -> str | None:
        if len(self.context) < 3:
            return None
        recent = "\n---\n".join(e.text for e in self.context[-10:])
        prompt = CONVENTION_PROMPT.format(experience=recent)
        answer = invoke_with_retry(self.reasoning_llm, [HumanMessage(content=prompt)])
        if answer.lower().startswith("none") or len(answer) < 10:
            return None
        return answer[:250]

    # ---- Reproduction readiness ----

    def should_reproduce_periodic(self) -> bool:
        return self.interactions >= self.config.interactions_per_lifetime

    def context_novelty(self) -> float:
        """Jaccard distance between older and recent halves of the context.

        High novelty (close to 1.0) means the agent has been accumulating
        diverse new information — a signal that it may be ready to distill.
        """
        if len(self.context) < 6:
            return 0.0
        mid = len(self.context) // 2
        old_words: set[str] = set()
        for e in self.context[:mid]:
            old_words.update(w for w in e.text.lower().split() if len(w) > 3)
        new_words: set[str] = set()
        for e in self.context[mid:]:
            new_words.update(w for w in e.text.lower().split() if len(w) > 3)
        union = old_words | new_words
        if not union:
            return 0.0
        return 1.0 - len(old_words & new_words) / len(union)

    def should_reproduce_novelty(self) -> bool:
        if not self.config.reproduce_on_novelty:
            return False
        return self.context_novelty() >= self.config.novelty_threshold

    def can_have_more_children(self) -> bool:
        return self.children_birthed < self.config.max_children_per_agent
