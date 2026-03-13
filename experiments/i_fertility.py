"""Experiment I: Fertility Ablation — Optimal Reproduction Frequency.

Sweeps how often agents reproduce and compares fixed-interval strategies
against a novelty-based trigger (reproduce when Jaccard distance between
recent and older context exceeds a threshold).

Part 1 — Fixed-interval vs. novelty-based reproduction:
  success_only:  periodic reproduction disabled; only on success
  every_3:       reproduce every 3 interactions  (hyper-fertile)
  every_7:       reproduce every 7 interactions  (high fertility)
  every_15:      reproduce every 15 interactions (medium fertility)
  every_30:      reproduce every 30 interactions (low fertility)
  novelty_0.5:   reproduce when context novelty > 0.5 (low threshold)
  novelty_0.7:   reproduce when context novelty > 0.7 (medium threshold)
  novelty_0.9:   reproduce when context novelty > 0.9 (high threshold)

Part 2 — Environmental robustness:
  Re-runs the top fixed strategy and novelty strategies on a larger,
  harder graph to test whether novelty-based triggers generalize.

Hypothesis: There is a sweet spot of fertility.  Too frequent reproduction
yields under-cooked priors; too rare wastes knowledge.  The novelty trigger
at threshold 0.7 approximates the sweet spot and generalises across
environments, unlike fixed-interval strategies which are tuned to a
specific graph difficulty.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import TrialConfig
from src.metrics import reproduction_stats, signal_precision, signal_recall
from src.runner import TrialRunner


CONDITIONS: dict[str, dict] = {
    "success_only": {
        "interactions_per_lifetime": 999,
        "reproduce_on_novelty": False,
    },
    "every_3": {
        "interactions_per_lifetime": 3,
        "reproduce_on_novelty": False,
    },
    "every_7": {
        "interactions_per_lifetime": 7,
        "reproduce_on_novelty": False,
    },
    "every_15": {
        "interactions_per_lifetime": 15,
        "reproduce_on_novelty": False,
    },
    "every_30": {
        "interactions_per_lifetime": 30,
        "reproduce_on_novelty": False,
    },
    "novelty_0.5": {
        "interactions_per_lifetime": 999,
        "reproduce_on_novelty": True,
        "novelty_threshold": 0.5,
    },
    "novelty_0.7": {
        "interactions_per_lifetime": 999,
        "reproduce_on_novelty": True,
        "novelty_threshold": 0.7,
    },
    "novelty_0.9": {
        "interactions_per_lifetime": 999,
        "reproduce_on_novelty": True,
        "novelty_threshold": 0.9,
    },
}

# Standard environment (small graph, used in Part 1)
BASE_ENV = {
    "num_nodes": 15,
    "num_doors": 4,
    "connection_radius": 0.40,
    "hints_per_door": 2,
    "distractors_per_door": 2,
    "min_goal_distance": 2,
}

# Robustness environment (larger, harder graph for Part 2)
ROBUST_ENV = {
    "num_nodes": 25,
    "num_doors": 6,
    "connection_radius": 0.35,
    "hints_per_door": 3,
    "distractors_per_door": 3,
    "min_goal_distance": 4,
}

# Conditions for Part 2: best fixed vs. novelty sweep
ROBUSTNESS_CONDITIONS: dict[str, dict] = {
    "every_15": CONDITIONS["every_15"],
    "every_30": CONDITIONS["every_30"],
    "novelty_0.5": CONDITIONS["novelty_0.5"],
    "novelty_0.7": CONDITIONS["novelty_0.7"],
    "novelty_0.9": CONDITIONS["novelty_0.9"],
}


def _run_sweep(
    conditions: dict[str, dict],
    env_params: dict,
    label: str,
    num_trials: int,
    seed_start: int,
    output: Path,
    overrides: dict,
) -> dict[str, list[dict]]:
    """Run a sweep of conditions on a given environment."""
    all_results: dict[str, list[dict]] = {c: [] for c in conditions}

    for cond_name, cond_overrides in conditions.items():
        print(f"\n{'='*60}")
        print(f"  Experiment I ({label}) — fertility: {cond_name}")
        print(f"{'='*60}")

        for trial_idx in range(num_trials):
            seed = seed_start + trial_idx
            merged = {**env_params, **overrides, **cond_overrides}
            config = TrialConfig(
                random_seed=seed,
                inherit_prior=True,
                reproduce_on_success=True,
                max_children_per_agent=6,
                success_count=4,
                max_steps=merged.pop("max_steps", 300),
                num_root_agents=2,
                log_transcript=True,
                **merged,
            )
            print(f"  Trial {trial_idx+1}/{num_trials} (seed={seed})...",
                  end=" ", flush=True)

            transcript_dir = output / "transcripts" / label / cond_name
            runner = TrialRunner(config, transcript_dir=transcript_dir)
            result = runner.run()

            trial_data = _extract_trial_data(
                seed, cond_name, result)
            all_results[cond_name].append(trial_data)
            triggers = trial_data["birth_triggers"]
            print(f"steps={result.total_steps}, "
                  f"success={len(result.successful_agents)}, "
                  f"births={trial_data['num_births']} "
                  f"({', '.join(f'{k}={v}' for k, v in triggers.items())})")

    return all_results


def _extract_trial_data(seed: int, cond_name: str, result) -> dict:
    repro = reproduction_stats(result.lineage.births)
    prec = [signal_precision(p, result.all_signals)
            for p in result.priors.values() if p]
    rec = [signal_recall(p, result.all_signals)
           for p in result.priors.values() if p]

    gen_success: dict[int, list[bool]] = {}
    for aid, data in result.per_agent_data.items():
        gen = data["generation"]
        gen_success.setdefault(gen, []).append(
            aid in result.successful_agents)
    gen_rates = {g: sum(v) / len(v)
                 for g, v in sorted(gen_success.items())}

    return {
        "seed": seed,
        "condition": cond_name,
        "total_steps": result.total_steps,
        "num_successful": len(result.successful_agents),
        "stopped_reason": result.stopped_reason,
        "num_births": repro["total_births"],
        "max_generation": repro["max_generation"],
        "birth_triggers": repro["by_trigger"],
        "avg_prior_precision": (sum(prec) / len(prec) if prec else 0),
        "avg_prior_recall": (sum(rec) / len(rec) if rec else 0),
        "gen_success_rates": gen_rates,
        "lineage_tree": result.lineage.tree_str(
            set(result.successful_agents)),
    }


def run(num_trials: int = 3, output_dir: str = "results/exp_i",
        seed_start: int = 900, **overrides) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Part 1: Full sweep on standard environment
    all_results = _run_sweep(
        CONDITIONS, BASE_ENV, "standard", num_trials, seed_start, output,
        overrides)

    summary = _summarize(all_results)
    summary["experiment"] = "I: Fertility Ablation"
    summary["hypothesis"] = (
        "A sweet spot of reproduction frequency produces better-performing "
        "children via higher-quality knowledge distillation."
    )

    (output / "results.json").write_text(
        json.dumps(all_results, indent=2, default=str))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    # Part 2: Robustness check on harder graph
    robust_results = _run_sweep(
        ROBUSTNESS_CONDITIONS, ROBUST_ENV, "robust",
        num_trials, seed_start + 100, output, overrides)

    robust_summary = _summarize(robust_results)
    robust_summary["experiment"] = "I-robust: Fertility Robustness"
    robust_summary["hypothesis"] = (
        "Novelty-based reproduction generalises across environments "
        "better than fixed-interval strategies."
    )

    (output / "robust_results.json").write_text(
        json.dumps(robust_results, indent=2, default=str))
    (output / "robust_summary.json").write_text(
        json.dumps(robust_summary, indent=2))

    report = _report(summary, all_results)
    report += "\n\n" + _report(robust_summary, robust_results)
    (output / "report.txt").write_text(report)
    print(f"\n{report}")
    return summary


def _summarize(results: dict[str, list[dict]]) -> dict:
    summary: dict = {}
    for cond, trials in results.items():
        steps = [t["total_steps"] for t in trials]
        births = [t["num_births"] for t in trials]
        prec = [t["avg_prior_precision"] for t in trials]
        rec = [t["avg_prior_recall"] for t in trials]
        max_gens = [t["max_generation"] for t in trials]
        success_rate = (sum(1 for t in trials if t["stopped_reason"] == "success")
                        / len(trials)) if trials else 0
        # Aggregate generation success rates
        all_gen: dict[int, list[float]] = {}
        for t in trials:
            for g, r in t.get("gen_success_rates", {}).items():
                all_gen.setdefault(int(g), []).append(r)
        avg_gen = {g: sum(v) / len(v) for g, v in sorted(all_gen.items())}

        summary[cond] = {
            "mean_steps": sum(steps) / len(steps) if steps else 0,
            "success_rate": success_rate,
            "mean_births": sum(births) / len(births) if births else 0,
            "mean_max_gen": sum(max_gens) / len(max_gens) if max_gens else 0,
            "mean_precision": sum(prec) / len(prec) if prec else 0,
            "mean_recall": sum(rec) / len(rec) if rec else 0,
            "gen_success": avg_gen,
            "n_trials": len(trials),
        }
    return summary


def _report(summary: dict, results: dict) -> str:
    lines = [
        "=" * 74,
        "  EXPERIMENT I: FERTILITY ABLATION",
        "=" * 74,
        "",
        f"Hypothesis: {summary.get('hypothesis', '')}",
        "",
        f"{'Condition':<16} {'Steps':<8} {'Success':<10} {'Births':<8} "
        f"{'MaxGen':<8} {'Prec':<8} {'Recall':<8}",
        "-" * 66,
    ]
    order = ["success_only", "every_3", "every_7", "every_15",
             "every_30", "novelty_0.5", "novelty_0.7", "novelty_0.9"]
    for cond in order:
        if cond in summary:
            s = summary[cond]
            lines.append(
                f"{cond:<16} {s['mean_steps']:<8.0f} "
                f"{s['success_rate']:<10.0%} {s['mean_births']:<8.1f} "
                f"{s['mean_max_gen']:<8.1f} "
                f"{s['mean_precision']:<8.3f} {s['mean_recall']:<8.3f}"
            )

    # Gen-0 vs Gen-1+ success comparison
    lines.extend(["", "  Child vs. root success rates:"])
    for cond in order:
        if cond in summary:
            gen_rates = summary[cond].get("gen_success", {})
            g0 = gen_rates.get(0, 0)
            g1_plus = [v for g, v in gen_rates.items() if g > 0]
            g1_avg = sum(g1_plus) / len(g1_plus) if g1_plus else 0
            lines.append(
                f"    {cond:<16} gen-0: {g0:5.0%}   "
                f"gen-1+: {g1_avg:5.0%}   "
                f"{'(children better)' if g1_avg > g0 else ''}"
            )

    lines.extend(["", "=" * 74])
    return "\n".join(lines)
