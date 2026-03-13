"""Experiment B: Do priors + interactions with parent agents help?

Three conditions:
  no_prior:          blank slate baseline
  prior_only:        compressed prior, no parent interaction
  prior_plus_query:  prior + 3 questions at steps 0, 3, 7

Measures: steps to goal, success rate, parent queries used, births.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import TrialConfig
from src.metrics import reproduction_stats
from src.runner import TrialRunner


def run(num_trials: int = 5, output_dir: str = "results/exp_b", seed_start: int = 200,
        **overrides) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    conditions = {
        "no_prior": {
            "inherit_prior": False,
            "enable_parent_query": False,
        },
        "prior_only": {
            "inherit_prior": True,
            "enable_parent_query": False,
        },
        "prior_plus_query": {
            "inherit_prior": True,
            "enable_parent_query": True,
            "max_parent_queries": 3,
            "parent_query_steps": (0, 3, 7),
        },
    }

    all_results: dict[str, list[dict]] = {c: [] for c in conditions}

    for cond_name, cond_overrides in conditions.items():
        print(f"\n{'='*60}")
        print(f"  Experiment B — condition: {cond_name}")
        print(f"{'='*60}")

        for trial_idx in range(num_trials):
            seed = seed_start + trial_idx
            merged = {**overrides, **cond_overrides}
            config = TrialConfig(random_seed=seed, log_transcript=True, **merged)
            print(f"  Trial {trial_idx+1}/{num_trials} (seed={seed})...", end=" ", flush=True)

            transcript_dir = output / "transcripts" / cond_name
            runner = TrialRunner(config, transcript_dir=transcript_dir)
            result = runner.run()

            total_queries = sum(
                d.get("parent_queries_used", 0) for d in result.per_agent_data.values()
            )
            repro = reproduction_stats(result.lineage.births)

            trial_data = {
                "seed": seed,
                "condition": cond_name,
                "total_steps": result.total_steps,
                "num_successful": len(result.successful_agents),
                "stopped_reason": result.stopped_reason,
                "num_births": repro["total_births"],
                "max_generation": repro["max_generation"],
                "total_parent_queries": total_queries,
                "lineage_tree": result.lineage.tree_str(set(result.successful_agents)),
                "priors": result.priors,
            }
            all_results[cond_name].append(trial_data)
            print(f"steps={result.total_steps}, success={len(result.successful_agents)}, "
                  f"births={repro['total_births']}, queries={total_queries}")

    summary = _summarize(all_results)
    summary["experiment"] = "B: Parent Interaction"
    summary["hypothesis"] = (
        "Priors + multi-step parent interaction outperform priors alone."
    )

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
        queries = [t.get("total_parent_queries", 0) for t in trials]
        births = [t["num_births"] for t in trials]
        success_rate = (sum(1 for t in trials if t["stopped_reason"] == "success")
                        / len(trials)) if trials else 0
        summary[cond] = {
            "mean_steps": sum(steps) / len(steps) if steps else 0,
            "mean_successes": sum(successes) / len(successes) if successes else 0,
            "success_rate": success_rate,
            "mean_queries": sum(queries) / len(queries) if queries else 0,
            "mean_births": sum(births) / len(births) if births else 0,
            "n_trials": len(trials),
        }
    return summary


def _report(summary: dict, results: dict) -> str:
    lines = [
        "=" * 70,
        "  EXPERIMENT B: PARENT INTERACTION",
        "=" * 70,
        "",
        f"Hypothesis: {summary.get('hypothesis', '')}",
        "",
        f"{'Condition':<20} {'Steps':<10} {'Success':<12} {'Births':<8} {'Queries':<8}",
        "-" * 58,
    ]
    for cond in ["no_prior", "prior_only", "prior_plus_query"]:
        if cond in summary:
            s = summary[cond]
            lines.append(
                f"{cond:<20} {s['mean_steps']:<10.1f} {s['success_rate']:<12.2%} "
                f"{s['mean_births']:<8.1f} {s.get('mean_queries', 0):<8.1f}"
            )
    lines.extend(["", "=" * 70])
    return "\n".join(lines)
