"""Experiment H: Cloaked Goals — Can LLM Agents Overcome Mathematically-Hidden Targets?

Uses potential-theory active cloaking (DeGiovanni & Guevara Vasquez, 2025) to
create environments where hints about the goal are attenuated by a
DTN-operator overlay.  Agents far from the goal see mostly noise; only by
penetrating the cloaking boundary do they start receiving real hints.

Conditions:
  uncloaked:        Standard environment, no prior
  uncloaked_prior:  Standard environment, prior inheritance
  cloaked:          Cloaked environment, no prior
  cloaked_prior:    Cloaked environment, prior inheritance
  cross_cloak:      Prior from UNCLOAKED world transferred to CLOAKED world

Hypothesis: Cloaking significantly increases difficulty.  Priors trained in
cloaked environments should be more abstract and transferable.  Cross-cloak
priors (from uncloaked) may partially help because they carry some
general strategy, but less than same-world priors.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import TrialConfig
from src.metrics import reproduction_stats, signal_precision, signal_recall
from src.runner import TrialRunner


def _collect_priors_from_trial(config_overrides: dict, seed: int) -> list[str]:
    """Run a short teacher trial and collect priors from experienced agents."""
    teacher_config = TrialConfig(
        random_seed=seed,
        success_count=2,
        num_root_agents=2,
        inherit_prior=True,
        reproduce_on_success=True,
        log_transcript=False,
        **config_overrides,
    )
    runner = TrialRunner(teacher_config)
    result = runner.run()
    priors = [p for p in result.priors.values() if p and len(p) > 20]
    for agent in runner.parent_refs.values():
        p = agent.compress_to_prior()
        if p and len(p) > 20 and p not in priors:
            priors.append(p)
    return priors[:4]


def run(num_trials: int = 3, output_dir: str = "results/exp_h",
        seed_start: int = 800, **overrides) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    base_env = {
        "num_nodes": 25,
        "num_doors": 5,
        "connection_radius": 0.35,
        "hints_per_door": 3,
        "distractors_per_door": 2,
        "min_goal_distance": 3,
    }

    conditions = {
        "uncloaked": {
            "enable_cloaking": False,
            "inherit_prior": False,
            "inject_priors": False,
            "teacher_cloaked": False,
        },
        "uncloaked_prior": {
            "enable_cloaking": False,
            "inherit_prior": True,
            "inject_priors": False,
            "teacher_cloaked": False,
        },
        "cloaked": {
            "enable_cloaking": True,
            "inherit_prior": False,
            "inject_priors": False,
            "teacher_cloaked": False,
        },
        "cloaked_prior": {
            "enable_cloaking": True,
            "inherit_prior": True,
            "inject_priors": False,
            "teacher_cloaked": False,
        },
        "cross_cloak": {
            "enable_cloaking": True,
            "inherit_prior": True,
            "inject_priors": True,
            "teacher_cloaked": False,
        },
    }

    all_results: dict[str, list[dict]] = {c: [] for c in conditions}

    for cond_name, cond_params in conditions.items():
        print(f"\n{'='*60}")
        print(f"  Experiment H — condition: {cond_name}")
        print(f"{'='*60}")

        for trial_idx in range(num_trials):
            seed = seed_start + trial_idx
            trial_overrides = {
                k: v for k, v in overrides.items()
                if k not in ("inject_priors", "teacher_cloaked",
                             "enable_cloaking", "inherit_prior")
            }
            config = TrialConfig(
                random_seed=seed,
                enable_cloaking=cond_params["enable_cloaking"],
                inherit_prior=cond_params["inherit_prior"],
                reproduce_on_success=True,
                log_transcript=True,
                **base_env,
                **trial_overrides,
            )
            print(f"  Trial {trial_idx+1}/{num_trials} (seed={seed})...",
                  end=" ", flush=True)

            initial_priors: list[str] = []
            if cond_params["inject_priors"]:
                teacher_env = {**base_env, "enable_cloaking": False}
                priors_from_uncloaked = _collect_priors_from_trial(
                    teacher_env, seed + 5000)
                initial_priors = priors_from_uncloaked[:config.num_root_agents]
                print(f"[{len(initial_priors)} cross-world priors] ",
                      end="", flush=True)

            transcript_dir = output / "transcripts" / cond_name
            runner = TrialRunner(config, transcript_dir=transcript_dir,
                                 initial_priors=initial_priors)
            result = runner.run()

            repro = reproduction_stats(result.lineage.births)
            prec = [signal_precision(p, result.all_signals)
                    for p in result.priors.values()]
            rec = [signal_recall(p, result.all_signals)
                   for p in result.priors.values()]

            cloak_info = {}
            if result.config.get("enable_cloaking"):
                for state in runner.active:
                    co = state.env.cloaking_overlay
                    if co:
                        cloak_info = {
                            "omega_size": len(co.omega_nodes),
                            "partial_omega_size": len(co.partial_omega_nodes),
                            "cloaking_metric": co.cloaking_metric,
                            "agent_start_visibility": co.signal_visibility.get(
                                state.env.agent_node, 1.0),
                        }
                        break

            trial_data = {
                "seed": seed,
                "condition": cond_name,
                "total_steps": result.total_steps,
                "num_successful": len(result.successful_agents),
                "stopped_reason": result.stopped_reason,
                "num_births": repro["total_births"],
                "max_generation": repro["max_generation"],
                "avg_prior_precision": (sum(prec) / len(prec) if prec else 0),
                "avg_prior_recall": (sum(rec) / len(rec) if rec else 0),
                "cloaking_info": cloak_info,
                "injected_priors": initial_priors[:2],
                "lineage_tree": result.lineage.tree_str(
                    set(result.successful_agents)),
                "priors": result.priors,
            }
            all_results[cond_name].append(trial_data)
            clk_tag = (f"Δ={cloak_info.get('cloaking_metric', 0):.3f} "
                       if cloak_info else "")
            print(f"steps={result.total_steps}, "
                  f"success={len(result.successful_agents)}, "
                  f"births={repro['total_births']} {clk_tag}")

    summary = _summarize(all_results)
    summary["experiment"] = "H: Cloaked Goals (Potential-Theory Signal Attenuation)"
    summary["hypothesis"] = (
        "Cloaking significantly increases difficulty. Priors help overcome "
        "cloaking, and cross-cloak priors from uncloaked worlds partially transfer."
    )

    (output / "results.json").write_text(
        json.dumps(all_results, indent=2, default=str))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    report = _report(summary, all_results)
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
        success_rate = (sum(1 for t in trials if t["stopped_reason"] == "success")
                        / len(trials)) if trials else 0
        summary[cond] = {
            "mean_steps": sum(steps) / len(steps) if steps else 0,
            "success_rate": success_rate,
            "mean_births": sum(births) / len(births) if births else 0,
            "mean_precision": sum(prec) / len(prec) if prec else 0,
            "mean_recall": sum(rec) / len(rec) if rec else 0,
            "n_trials": len(trials),
        }
    return summary


def _report(summary: dict, results: dict) -> str:
    lines = [
        "=" * 74,
        "  EXPERIMENT H: CLOAKED GOALS (POTENTIAL-THEORY SIGNAL ATTENUATION)",
        "=" * 74,
        "",
        f"Hypothesis: {summary.get('hypothesis', '')}",
        "",
        f"{'Condition':<20} {'Steps':<8} {'Success':<10} {'Births':<8} "
        f"{'Prec':<8} {'Recall':<8}",
        "-" * 62,
    ]
    order = ["uncloaked", "uncloaked_prior", "cloaked", "cloaked_prior",
             "cross_cloak"]
    for cond in order:
        if cond in summary:
            s = summary[cond]
            lines.append(
                f"{cond:<20} {s['mean_steps']:<8.0f} "
                f"{s['success_rate']:<10.0%} {s['mean_births']:<8.1f} "
                f"{s['mean_precision']:<8.3f} {s['mean_recall']:<8.3f}"
            )

    # Cloaking metrics
    for cond in ["cloaked", "cloaked_prior", "cross_cloak"]:
        if cond in results:
            for t in results[cond]:
                ci = t.get("cloaking_info", {})
                if ci:
                    lines.extend([
                        "",
                        f"  Cloaking info ({cond}, seed={t['seed']}):",
                        f"    Ω size:          {ci.get('omega_size', 0)}",
                        f"    ∂Ω size:         {ci.get('partial_omega_size', 0)}",
                        f"    Δ_cloak:         {ci.get('cloaking_metric', 0):.4f}",
                        f"    Start visibility:{ci.get('agent_start_visibility', 0):.3f}",
                    ])
                    break

    # Cross-cloak prior samples
    if "cross_cloak" in results:
        for t in results["cross_cloak"]:
            if t.get("injected_priors"):
                lines.extend(["", "  Cross-cloak injected priors:"])
                for i, p in enumerate(t["injected_priors"]):
                    lines.append(f"    [{i+1}] {p[:140]}...")
                break

    # Comparison summary
    uncloaked = summary.get("uncloaked", {})
    cloaked = summary.get("cloaked", {})
    if uncloaked and cloaked:
        u_steps = uncloaked.get("mean_steps", 0)
        c_steps = cloaked.get("mean_steps", 0)
        if u_steps > 0:
            slowdown = c_steps / u_steps
            lines.extend([
                "",
                f"  Cloaking slowdown: {slowdown:.1f}x "
                f"({u_steps:.0f} → {c_steps:.0f} steps)",
            ])
            u_rate = uncloaked.get("success_rate", 0)
            c_rate = cloaked.get("success_rate", 0)
            lines.append(
                f"  Success rate drop: {u_rate:.0%} → {c_rate:.0%}"
            )

    lines.extend(["", "=" * 74])
    return "\n".join(lines)
