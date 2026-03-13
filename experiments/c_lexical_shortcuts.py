"""Experiment C: Lexical shortcuts and convention formation.

Tests whether agents develop stable shorthand across generations.
Varies environment complexity.

Three scales:
  small:  10 nodes, 3 doors, 1 hint, 1 distractor
  medium: 15 nodes, 4 doors, 2 hints, 2 distractors
  large:  25 nodes, 6 doors, 3 hints, 4 distractors
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import TrialConfig
from src.metrics import (
    compute_drift_chain,
    extract_frequent_phrases,
    reproduction_stats,
)
from src.runner import TrialRunner


ENV_SCALES = {
    "small": {
        "num_nodes": 10, "connection_radius": 0.5,
        "num_doors": 3, "hints_per_door": 1, "distractors_per_door": 1,
        "min_goal_distance": 2,
    },
    "medium": {
        "num_nodes": 15, "connection_radius": 0.4,
        "num_doors": 4, "hints_per_door": 2, "distractors_per_door": 2,
        "min_goal_distance": 3,
    },
    "large": {
        "num_nodes": 25, "connection_radius": 0.35,
        "num_doors": 6, "hints_per_door": 3, "distractors_per_door": 4,
        "min_goal_distance": 4,
    },
}


def run(num_trials: int = 3, output_dir: str = "results/exp_c", seed_start: int = 300,
        **overrides) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}

    for scale_name, scale_params in ENV_SCALES.items():
        print(f"\n{'='*60}")
        print(f"  Experiment C — environment scale: {scale_name}")
        print(f"{'='*60}")
        all_results[scale_name] = []

        for trial_idx in range(num_trials):
            seed = seed_start + trial_idx
            merged = {**scale_params, **overrides}
            config = TrialConfig(
                random_seed=seed, inherit_prior=True, log_transcript=True, **merged,
            )
            print(f"  Trial {trial_idx+1}/{num_trials} (seed={seed})...", end=" ", flush=True)

            transcript_dir = output / "transcripts" / scale_name
            runner = TrialRunner(config, transcript_dir=transcript_dir)
            result = runner.run()

            drift_data = _analyze_lineage_drift(result)
            all_priors = list(result.priors.values())
            phrases = extract_frequent_phrases(all_priors, min_count=2, n=3)
            repro = reproduction_stats(result.lineage.births)

            trial_data = {
                "seed": seed,
                "scale": scale_name,
                "total_steps": result.total_steps,
                "num_successful": len(result.successful_agents),
                "stopped_reason": result.stopped_reason,
                "num_births": repro["total_births"],
                "max_generation": repro["max_generation"],
                "drift_chains": drift_data,
                "common_phrases": phrases,
                "num_unique_priors": len(set(result.priors.values())),
                "lineage_tree": result.lineage.tree_str(set(result.successful_agents)),
            }
            all_results[scale_name].append(trial_data)
            print(f"steps={result.total_steps}, births={repro['total_births']}, "
                  f"conventions={len(phrases)}")

    summary = _summarize(all_results)
    summary["experiment"] = "C: Lexical Shortcuts & Convention Formation"
    summary["hypothesis"] = (
        "Agents develop stable shorthand conventions. "
        "Larger environments produce more drift."
    )

    (output / "results.json").write_text(json.dumps(all_results, indent=2, default=str))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    report = _report(summary, all_results)
    (output / "report.txt").write_text(report)
    print(f"\n{report}")

    return summary


def _analyze_lineage_drift(result) -> list[dict]:
    births = result.lineage.births
    parents = result.lineage.parents
    priors = result.priors

    roots = [a for a, p in parents.items() if p is None]
    chains = []
    for root in roots:
        chain = [root]
        current = root
        while True:
            children = [c for c, p in parents.items() if p == current]
            if not children:
                break
            current = children[0]
            chain.append(current)
        if len(chain) >= 2:
            prior_chain = [priors.get(a, "") for a in chain if a in priors]
            if len(prior_chain) >= 2:
                drift = compute_drift_chain(prior_chain)
                chains.append({"agents": chain, "drift": drift})

    return chains


def _summarize(results: dict[str, list[dict]]) -> dict:
    summary = {}
    for scale, trials in results.items():
        all_drift = []
        all_phrases = []
        for t in trials:
            for chain in t.get("drift_chains", []):
                all_drift.extend(chain.get("drift", []))
            all_phrases.extend(t.get("common_phrases", []))

        avg_jaccard = (sum(d["jaccard"] for d in all_drift) / len(all_drift)) if all_drift else 0
        avg_novelty = (sum(d["ngram_novelty"] for d in all_drift) / len(all_drift)) if all_drift else 0

        summary[scale] = {
            "mean_steps": sum(t["total_steps"] for t in trials) / len(trials) if trials else 0,
            "mean_births": sum(t["num_births"] for t in trials) / len(trials) if trials else 0,
            "avg_parent_child_jaccard": avg_jaccard,
            "avg_ngram_novelty": avg_novelty,
            "num_common_phrases": len(set(p[0] for p in all_phrases)),
            "n_trials": len(trials),
        }
    return summary


def _report(summary: dict, results: dict) -> str:
    lines = [
        "=" * 70,
        "  EXPERIMENT C: LEXICAL SHORTCUTS & CONVENTION FORMATION",
        "=" * 70,
        "",
        f"Hypothesis: {summary.get('hypothesis', '')}",
        "",
        f"{'Scale':<10} {'Steps':<10} {'Births':<8} {'Jaccard':<10} {'Novelty':<10} {'Conventions':<12}",
        "-" * 60,
    ]
    for scale in ["small", "medium", "large"]:
        if scale in summary:
            s = summary[scale]
            lines.append(
                f"{scale:<10} {s['mean_steps']:<10.0f} {s['mean_births']:<8.1f} "
                f"{s['avg_parent_child_jaccard']:<10.3f} {s['avg_ngram_novelty']:<10.3f} "
                f"{s['num_common_phrases']:<12}"
            )

    for scale, trials in results.items():
        all_phrases = []
        for t in trials:
            all_phrases.extend(t.get("common_phrases", []))
        if all_phrases:
            lines.extend(["", f"  Common phrases ({scale}):"])
            seen = set()
            for phrase, count in sorted(all_phrases, key=lambda x: -x[1]):
                if phrase not in seen:
                    lines.append(f"    '{phrase}' (x{count})")
                    seen.add(phrase)
                if len(seen) >= 5:
                    break

    lines.extend(["", "=" * 70])
    return "\n".join(lines)
