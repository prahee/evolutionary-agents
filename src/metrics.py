"""Metrics for measuring drift, similarity, signal quality, beliefs, and conventions."""

from __future__ import annotations

import math
from collections import Counter


# ---------------------------------------------------------------------------
# Text similarity
# ---------------------------------------------------------------------------

def jaccard_similarity(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def word_overlap_ratio(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)


def ngram_novelty(parent_text: str, child_text: str, n: int = 3) -> float:
    def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
        words = text.lower().split()
        return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

    parent_ng = ngrams(parent_text, n)
    child_ng = ngrams(child_text, n)
    if not child_ng:
        return 0.0
    novel = child_ng - parent_ng
    return len(novel) / len(child_ng)


def edit_distance(a: str, b: str) -> int:
    wa = a.lower().split()
    wb = b.lower().split()
    m, n = len(wa), len(wb)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if wa[i-1] == wb[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]


# ---------------------------------------------------------------------------
# Signal analysis
# ---------------------------------------------------------------------------

def signal_precision(prior_text: str, all_signals: list[tuple[str, bool]]) -> float:
    prior_lower = prior_text.lower()
    referenced_hints = 0
    referenced_total = 0
    for text, is_hint in all_signals:
        key_words = set(text.lower().split()) - {"the", "a", "is", "to", "in", "of", "and"}
        if any(w in prior_lower for w in key_words if len(w) > 3):
            referenced_total += 1
            if is_hint:
                referenced_hints += 1
    return referenced_hints / referenced_total if referenced_total > 0 else 0.0


def signal_recall(prior_text: str, all_signals: list[tuple[str, bool]]) -> float:
    prior_lower = prior_text.lower()
    total_hints = sum(1 for _, h in all_signals if h)
    if total_hints == 0:
        return 0.0
    recalled = 0
    for text, is_hint in all_signals:
        if not is_hint:
            continue
        key_words = set(text.lower().split()) - {"the", "a", "is", "to", "in", "of", "and"}
        if any(w in prior_lower for w in key_words if len(w) > 3):
            recalled += 1
    return recalled / total_hints


# ---------------------------------------------------------------------------
# Prior analysis
# ---------------------------------------------------------------------------

def prior_token_count(text: str) -> int:
    return len(text.split())


def extract_frequent_phrases(
    texts: list[str], min_count: int = 2, n: int = 3,
) -> list[tuple[str, int]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            counter[tuple(words[i:i+n])] += 1
    return [(" ".join(ng), c) for ng, c in counter.most_common(20) if c >= min_count]


# ---------------------------------------------------------------------------
# Drift between generations
# ---------------------------------------------------------------------------

def compute_drift_chain(priors_by_generation: list[str]) -> list[dict]:
    if len(priors_by_generation) < 2:
        return []
    results = []
    for i in range(1, len(priors_by_generation)):
        parent = priors_by_generation[i - 1]
        child = priors_by_generation[i]
        results.append({
            "generation": i,
            "jaccard": jaccard_similarity(parent, child),
            "word_overlap": word_overlap_ratio(parent, child),
            "ngram_novelty": ngram_novelty(parent, child),
            "edit_distance": edit_distance(parent, child),
            "parent_tokens": prior_token_count(parent),
            "child_tokens": prior_token_count(child),
        })
    return results


# ---------------------------------------------------------------------------
# Belief metrics
# ---------------------------------------------------------------------------

def belief_accuracy_over_time(trajectories: list[dict]) -> dict:
    """Summarize belief trajectory for a single agent."""
    if not trajectories:
        return {}
    return {
        "final_goal_belief": trajectories[-1].get("goal_belief", 0),
        "final_entropy": trajectories[-1].get("entropy", 0),
        "max_goal_belief": max(t.get("goal_belief", 0) for t in trajectories),
        "steps_to_correct_map": next(
            (t["step"] for t in trajectories if t.get("map_correct")), -1
        ),
        "fraction_correct_map": (
            sum(1 for t in trajectories if t.get("map_correct")) / len(trajectories)
        ),
    }


def avg_belief_metrics(all_trajectories: dict[str, list[dict]]) -> dict:
    """Average belief metrics across all agents in a trial."""
    if not all_trajectories:
        return {}
    metrics = [belief_accuracy_over_time(t) for t in all_trajectories.values() if t]
    if not metrics:
        return {}
    keys = ["final_goal_belief", "final_entropy", "max_goal_belief", "fraction_correct_map"]
    return {
        k: sum(m.get(k, 0) for m in metrics) / len(metrics)
        for k in keys
    }


# ---------------------------------------------------------------------------
# Reproduction analysis
# ---------------------------------------------------------------------------

def reproduction_stats(lineage_births: list) -> dict:
    """Analyze reproduction patterns from birth events."""
    if not lineage_births:
        return {"total_births": 0, "by_trigger": {}, "max_generation": 0}
    triggers: Counter = Counter()
    max_gen = 0
    for b in lineage_births:
        triggers[b.trigger] += 1
        max_gen = max(max_gen, b.generation)
    return {
        "total_births": len(lineage_births),
        "by_trigger": dict(triggers),
        "max_generation": max_gen,
    }
