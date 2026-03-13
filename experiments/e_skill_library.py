"""Experiment E: Cumulative knowledge via skill libraries.

Three conditions:
  no_prior_no_lib:    pure baseline
  prior_only:         prior inheritance, no shared library
  prior_plus_library: prior + shared categorized skill library

Measures: steps to goal, library growth, convention quality, per-generation success.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import TrialConfig
from src.metrics import reproduction_stats
from src.runner import TrialRunner


def run(num_trials: int = 5, output_dir: str = "results/exp_e", seed_start: int = 500,
        **overrides) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    conditions = {
        "no_prior_no_lib": {
            "inherit_prior": False,
            "enable_skill_library": False,
        },
        "prior_only": {
            "inherit_prior": True,
            "enable_skill_library": False,
        },
        "prior_plus_library": {
            "inherit_prior": True,
            "enable_skill_library": True,
        },
    }

    all_results: dict[str, list[dict]] = {c: [] for c in conditions}

    for cond_name, cond_overrides in conditions.items():
        print(f"\n{'='*60}")
        print(f"  Experiment E — condition: {cond_name}")
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
            lib_snapshot = result.skill_library_snapshot
            lib_size = len(lib_snapshot)

            lib_top = []
            if lib_snapshot:
                sorted_convs = sorted(lib_snapshot, key=lambda c: c["verified_count"], reverse=True)
                lib_top = [
                    f"[{c['category']}] {c['text']} (verified={c['verified_count']})"
                    for c in sorted_convs[:8]
                ]

            lib_stats = {}
            if lib_snapshot:
                cats: dict[str, int] = {}
                total_q = 0
                for c in lib_snapshot:
                    cats[c["category"]] = cats.get(c["category"], 0) + 1
                    total_q += c["times_queried"]
                lib_stats = {"size": lib_size, "categories": cats, "total_queries": total_q}

            gen_success: dict[int, list[bool]] = {}
            for aid, data in result.per_agent_data.items():
                gen = data["generation"]
                succeeded = aid in result.successful_agents
                gen_success.setdefault(gen, []).append(succeeded)
            gen_rates = {g: sum(v) / len(v) for g, v in sorted(gen_success.items())}

            trial_data = {
                "seed": seed,
                "condition": cond_name,
                "total_steps": result.total_steps,
                "num_successful": len(result.successful_agents),
                "stopped_reason": result.stopped_reason,
                "num_births": repro["total_births"],
                "max_generation": repro["max_generation"],
                "library_size": lib_size,
                "library_stats": lib_stats,
                "library_top_conventions": lib_top,
                "generation_success_rates": gen_rates,
                "lineage_tree": result.lineage.tree_str(set(result.successful_agents)),
            }
            all_results[cond_name].append(trial_data)
            print(f"steps={result.total_steps}, success={len(result.successful_agents)}, "
                  f"births={repro['total_births']}, lib={lib_size}")

    summary = _summarize(all_results)
    summary["experiment"] = "E: Skill Library (Voyager-inspired)"
    summary["hypothesis"] = (
        "A shared skill library accelerates collective learning beyond prior inheritance alone."
    )

    (output / "results.json").write_text(json.dumps(all_results, indent=2, default=str))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    report = _report(summary, all_results)
    (output / "report.txt").write_text(report)
    print(f"\n{report}")

    return summary


def _summarize(results: dict[str, list[dict]]) -> dict:
    summary = {}
    for cond, trials in results.items():
        steps = [t["total_steps"] for t in trials]
        successes = [t["num_successful"] for t in trials]
        lib_sizes = [t.get("library_size", 0) for t in trials]
        births = [t["num_births"] for t in trials]
        success_rate = (sum(1 for t in trials if t["stopped_reason"] == "success")
                        / len(trials)) if trials else 0

        all_gen_rates: dict[int, list[float]] = {}
        for t in trials:
            for g, r in t.get("generation_success_rates", {}).items():
                all_gen_rates.setdefault(int(g), []).append(r)
        avg_gen_rates = {g: sum(v) / len(v) for g, v in sorted(all_gen_rates.items())}

        total_queries = sum(
            t.get("library_stats", {}).get("total_queries", 0) for t in trials
        )
        all_cats: dict[str, int] = {}
        for t in trials:
            for cat, cnt in t.get("library_stats", {}).get("categories", {}).items():
                all_cats[cat] = all_cats.get(cat, 0) + cnt

        summary[cond] = {
            "mean_steps": sum(steps) / len(steps) if steps else 0,
            "mean_successes": sum(successes) / len(successes) if successes else 0,
            "success_rate": success_rate,
            "mean_lib_size": sum(lib_sizes) / len(lib_sizes) if lib_sizes else 0,
            "mean_births": sum(births) / len(births) if births else 0,
            "gen_success_rates": avg_gen_rates,
            "total_lib_queries": total_queries,
            "lib_categories": all_cats,
            "n_trials": len(trials),
        }
    return summary


def _report(summary: dict, results: dict) -> str:
    lines = [
        "=" * 70,
        "  EXPERIMENT E: SKILL LIBRARY (VOYAGER-INSPIRED)",
        "=" * 70,
        "",
        f"Hypothesis: {summary.get('hypothesis', '')}",
        "",
        f"{'Condition':<22} {'Steps':<8} {'Success':<10} {'Births':<8} {'Lib Size':<10}",
        "-" * 58,
    ]
    for cond in ["no_prior_no_lib", "prior_only", "prior_plus_library"]:
        if cond in summary:
            s = summary[cond]
            lines.append(
                f"{cond:<22} {s['mean_steps']:<8.0f} {s['success_rate']:<10.2%} "
                f"{s['mean_births']:<8.1f} {s.get('mean_lib_size', 0):<10.1f}"
            )

    lib_cond = "prior_plus_library"
    if lib_cond in summary and summary[lib_cond].get("lib_categories"):
        cats = summary[lib_cond]["lib_categories"]
        lines.extend(["", "  Convention categories:"])
        for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
            lines.append(f"    {cat}: {cnt}")

    if lib_cond in results:
        all_convs = []
        for t in results[lib_cond]:
            all_convs.extend(t.get("library_top_conventions", []))
        if all_convs:
            lines.extend(["", "  Top conventions:"])
            seen = set()
            for conv in all_convs:
                if conv not in seen:
                    lines.append(f"    {conv}")
                    seen.add(conv)
                if len(seen) >= 10:
                    break

    for cond in ["no_prior_no_lib", "prior_only", "prior_plus_library"]:
        if cond in summary and summary[cond].get("gen_success_rates"):
            rates = summary[cond]["gen_success_rates"]
            lines.extend(["", f"  Success by generation ({cond}):"])
            for g, r in sorted(rates.items()):
                bar = "#" * int(r * 20)
                lines.append(f"    Gen {g}: {r:5.0%} {bar}")

    lines.extend(["", "=" * 70])
    return "\n".join(lines)
