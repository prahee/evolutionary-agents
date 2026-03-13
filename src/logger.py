"""Transcript logger for recording all experiment events.

Produces a human-readable text report per trial that shows exactly
what each agent saw, thought, said, and passed to its children.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path


class TranscriptLogger:

    def __init__(self):
        self._buf = StringIO()
        self._step = 0

    def _write(self, text: str) -> None:
        self._buf.write(text)
        self._buf.write("\n")

    def _divider(self, char: str = "-", width: int = 76) -> None:
        self._write(char * width)

    def section(self, title: str) -> None:
        self._write("")
        self._divider("=")
        self._write(f"  {title}")
        self._divider("=")
        self._write("")

    def subsection(self, title: str) -> None:
        self._write("")
        self._divider("-", 60)
        self._write(f"  {title}")
        self._divider("-", 60)

    # ---- Environment ----

    def log_env_setup(self, description: str) -> None:
        self.section("ENVIRONMENT")
        self._write(description)

    # ---- Agent decisions ----

    def log_decision(
        self,
        agent_id: str,
        generation: int,
        step: int,
        observation_text: str,
        skill_library_context: str,
        llm_response: str,
        parsed_action: str,
        reasoning: str,
    ) -> None:
        self._step = step
        self.subsection(f"STEP {step} | {agent_id} (gen {generation})")
        self._write("OBSERVATION:")
        for line in observation_text.split("\n"):
            self._write(f"  {line}")
        if skill_library_context:
            self._write("")
            self._write("SKILL LIBRARY CONTEXT:")
            for line in skill_library_context.split("\n"):
                self._write(f"  {line}")
        self._write("")
        self._write("LLM RESPONSE (reasoning model):")
        for line in llm_response.split("\n"):
            self._write(f"  {line}")
        self._write("")
        self._write(f"PARSED ACTION: {parsed_action}")
        self._write(f"REASONING:     {reasoning}")

    # ---- Context summarisation ----

    def log_summarization(
        self,
        agent_id: str,
        num_entries_compressed: int,
        original_text_preview: str,
        summary: str,
    ) -> None:
        self.subsection(f"CONTEXT COMPRESSION | {agent_id}")
        self._write(f"Compressed {num_entries_compressed} context entries.")
        self._write("")
        self._write("ORIGINAL (preview):")
        for line in original_text_preview[:600].split("\n"):
            self._write(f"  {line}")
        if len(original_text_preview) > 600:
            self._write("  ...")
        self._write("")
        self._write("COMPRESSED TO:")
        for line in summary.split("\n"):
            self._write(f"  {line}")

    # ---- Prior compression ----

    def log_prior_compression(
        self,
        agent_id: str,
        generation: int,
        context_snapshot: str,
        compressed_prior: str,
    ) -> None:
        self.section(f"PRIOR COMPRESSION | {agent_id} (gen {generation})")
        self._write("FULL CONTEXT AT TIME OF COMPRESSION:")
        for line in context_snapshot.split("\n"):
            self._write(f"  {line}")
        self._write("")
        self._write("COMPRESSED PRIOR (passed to child):")
        for line in compressed_prior.split("\n"):
            self._write(f"  {line}")
        self._write(f"  [{len(compressed_prior.split())} words]")

    # ---- Birth ----

    def log_birth(
        self,
        parent_id: str,
        child_id: str,
        parent_generation: int,
        child_generation: int,
        step: int,
        child_prior: str,
        trigger: str = "periodic",
    ) -> None:
        self.section(
            f"BIRTH ({trigger}) | {parent_id} (gen {parent_generation}) "
            f"-> {child_id} (gen {child_generation}) at step {step}"
        )
        self._write("PRIOR INHERITED BY CHILD:")
        if child_prior:
            for line in child_prior.split("\n"):
                self._write(f"  {line}")
        else:
            self._write("  (no prior — blank slate)")

    # ---- Parent-child interaction ----

    def log_parent_query(
        self,
        child_id: str,
        parent_id: str,
        child_step: int,
        question: str,
        answer: str,
    ) -> None:
        self.subsection(
            f"PARENT QUERY | {child_id} asks {parent_id} (child step {child_step})"
        )
        self._write(f"QUESTION: {question}")
        self._write(f"ANSWER:   {answer}")

    # ---- Skill library ----

    def log_skill_library_add(
        self,
        agent_id: str,
        convention: str,
        accepted: bool,
    ) -> None:
        tag = "ADDED" if accepted else "DEDUPLICATED"
        self._write(f"  [SKILL LIBRARY {tag}] {agent_id}: \"{convention}\"")

    def log_skill_library_query(
        self,
        agent_id: str,
        returned_conventions: str,
    ) -> None:
        if returned_conventions:
            self._write(f"  [SKILL LIBRARY QUERY] {agent_id} received:")
            for line in returned_conventions.split("\n"):
                self._write(f"    {line}")

    # ---- Beliefs ----

    def log_belief_update(
        self,
        agent_id: str,
        step: int,
        belief_text: str,
        entropy: float,
        goal_belief: float,
    ) -> None:
        self._write(
            f"  [BELIEF] {agent_id} step {step}: "
            f"entropy={entropy:.2f}, P(goal)={goal_belief:.1%}"
        )

    # ---- Success ----

    def log_success(self, agent_id: str, generation: int, step: int) -> None:
        self.section(
            f"SUCCESS | {agent_id} (gen {generation}) found the goal at step {step}"
        )

    # ---- Final state ----

    def log_final_context(self, agent_id: str, context_snapshot: str) -> None:
        self.subsection(f"FINAL CONTEXT | {agent_id}")
        for line in context_snapshot.split("\n"):
            self._write(f"  {line}")

    def log_lineage_tree(self, tree_str: str) -> None:
        self.section("LINEAGE TREE")
        self._write(tree_str)

    # ---- Output ----

    def get_text(self) -> str:
        return self._buf.getvalue()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._buf.getvalue(), encoding="utf-8")

    def close(self) -> None:
        self._buf.close()
