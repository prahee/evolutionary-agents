"""Experiment A: Do priors given by parent exploration help?

Compares:
  inherited: children receive compressed prior from parent
  no_prior:  children start with empty prior (blank slate)

Measures: steps to goal, success rate, births, signal precision/recall in priors.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import TrialConfig
from src.metrics import reproduction_stats, signal_precision, signal_recall
from src.runner import TrialRunner


def run(num_trials: int = 5, output_dir: str = "results/exp_a", seed_start: int = 100,
        **overrides) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    conditions = {
        "inherited": {"inherit_prior": True},
        "no_prior": {"inherit_prior": False},
    }

    all_results: dict[str, list[dict]] = {c: [] for c in conditions}

    for cond_name, cond_overrides in conditions.items():
        print(f"\n{'='*60}")
        print(f"  Experiment A — condition: {cond_name}")
        print(f"{'='*60}")

        for trial_idx in range(num_trials):
            seed = seed_start + trial_idx
            merged = {**overrides, **cond_overrides}
            config = TrialConfig(random_seed=seed, log_transcript=True, **merged)
            print(f"  Trial {trial_idx+1}/{num_trials} (seed={seed})...", end=" ", flush=True)

            transcript_dir = output / "transcripts" / cond_name
            runner = TrialRunner(config, transcript_dir=transcript_dir)
            result = runner.run()

            repro = reproduction_stats(result.lineage.births)
            prior_precision = [signal_precision(p, result.all_signals) for p in result.priors.values()]
            prior_recall = [signal_recall(p, result.all_signals) for p in result.priors.values()]

            trial_data = {
                "seed": seed,
                "condition": cond_name,
                "total_steps": result.total_steps,
                "num_successful": len(result.successful_agents),
                "stopped_reason": result.stopped_reason,
                "num_births": repro["total_births"],
                "max_generation": repro["max_generation"],
                "birth_triggers": repro["by_trigger"],
                "avg_prior_precision": sum(prior_precision) / len(prior_precision) if prior_precision else 0,
                "avg_prior_recall": sum(prior_recall) / len(prior_recall) if prior_recall else 0,
                "lineage_tree": result.lineage.tree_str(set(result.successful_agents)),
                "priors": result.priors,
            }
            all_results[cond_name].append(trial_data)
            print(f"steps={result.total_steps}, success={len(result.successful_agents)}, "
                  f"births={repro['total_births']}")

    summary = _summarize(all_results)
    summary["experiment"] = "A: Prior Ablation"
    summary["hypothesis"] = "Agents with inherited priors reach the goal in fewer steps."

    (output / "results.json").write_text(json.dumps(all_results, indent=2, default=str))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    report = _report(summary, all_results)
    (output / "report.txt").write_text(report)
    print(f"\n{report}")

    return summary


def _summarize(results: dict[str, list[dict]]) -> dict:
    summary = {}
    for cond, trials in results.items():
        steps = [t["total_steps"] for t in trials]
        successes = [t["num_successful"] for t in trials]
        births = [t["num_births"] for t in trials]
        max_gens = [t.get("max_generation", 0) for t in trials]
        success_rate = (sum(1 for t in trials if t["stopped_reason"] == "success")
                        / len(trials)) if trials else 0
        summary[cond] = {
            "mean_steps": sum(steps) / len(steps) if steps else 0,
            "mean_successes": sum(successes) / len(successes) if successes else 0,
            "mean_births": sum(births) / len(births) if births else 0,
            "max_generation": max(max_gens) if max_gens else 0,
            "success_rate": success_rate,
            "n_trials": len(trials),
        }
    return summary


def _report(summary: dict, results: dict) -> str:
    lines = [
        "=" * 70,
        "  EXPERIMENT A: PRIOR ABLATION",
        "=" * 70,
        "",
        f"Hypothesis: {summary.get('hypothesis', '')}",
        "",
        f"{'Condition':<16} {'Steps':<10} {'Success':<12} {'Births':<8} {'Max Gen':<8}",
        "-" * 54,
    ]
    for cond in ["inherited", "no_prior"]:
        if cond in summary:
            s = summary[cond]
            lines.append(
                f"{cond:<16} {s['mean_steps']:<10.1f} {s['success_rate']:<12.2%} "
                f"{s['mean_births']:<8.1f} {s['max_generation']:<8}"
            )
    lines.extend(["", "=" * 70])
    return "\n".join(lines)
