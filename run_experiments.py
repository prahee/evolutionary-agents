"""CLI entry point for running the experiment suite.

Usage:
    uv run python run_experiments.py --exp a          # single experiment
    uv run python run_experiments.py --exp a c e i    # selected experiments
    uv run python run_experiments.py --exp all         # full suite
    uv run python run_experiments.py --exp a --trials 10 --max-steps 500
"""

from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv

from src.config import TrialConfig, validate_config_models


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Multi-agent POMDP experiment suite.")
    parser.add_argument(
        "--exp", nargs="+", default=["all"],
        help="Which experiments to run: a, b, c, e, h, i, or 'all'",
    )
    parser.add_argument("--trials", type=int, default=3, help="Trials per condition")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per trial")
    parser.add_argument(
        "--reasoning-model", type=str, default=None,
        help="Override reasoning model",
    )
    parser.add_argument(
        "--utility-model", type=str, default=None,
        help="Override utility model",
    )
    parser.add_argument("--output", type=str, default="results", help="Base output directory")
    parser.add_argument("--seed", type=int, default=100, help="Starting seed")
    args = parser.parse_args()

    probe = TrialConfig()
    if args.reasoning_model:
        probe.reasoning_model = args.reasoning_model
    if args.utility_model:
        probe.utility_model = args.utility_model
    validate_config_models(probe)

    common: dict = {"max_steps": args.max_steps}
    if args.reasoning_model:
        common["reasoning_model"] = args.reasoning_model
    if args.utility_model:
        common["utility_model"] = args.utility_model

    experiments = args.exp
    if "all" in experiments:
        experiments = ["a", "b", "c", "e", "h", "i"]

    print("=" * 70)
    print("  MULTI-AGENT POMDP EXPERIMENT SUITE")
    print("=" * 70)
    print(f"  Experiments:      {', '.join(experiments)}")
    print(f"  Trials/cond:      {args.trials}")
    print(f"  Max steps:        {args.max_steps}")
    print(f"  Reasoning model:  {probe.reasoning_model}")
    print(f"  Utility model:    {probe.utility_model}")
    print(f"  Output:           {args.output}/")
    print("=" * 70)

    exp_map = {
        "a": ("experiments.a_prior_ablation", 0),
        "b": ("experiments.b_parent_interaction", 100),
        "c": ("experiments.c_lexical_shortcuts", 200),
        "e": ("experiments.e_skill_library", 400),
        "h": ("experiments.h_cloaked_goals", 700),
        "i": ("experiments.i_fertility", 800),
    }

    for exp_id in experiments:
        if exp_id not in exp_map:
            print(f"  Unknown experiment: {exp_id}")
            continue

        module_name, seed_offset = exp_map[exp_id]
        t0 = time.time()
        print(f"\n{'#'*70}")
        print(f"  Starting experiment {exp_id.upper()}")
        print(f"{'#'*70}")

        import importlib
        mod = importlib.import_module(module_name)
        mod.run(
            num_trials=args.trials,
            output_dir=f"{args.output}/exp_{exp_id}",
            seed_start=args.seed + seed_offset,
            **common,
        )

        elapsed = time.time() - t0
        print(f"\n  Experiment {exp_id.upper()} completed in {elapsed:.1f}s")

    print(f"\n{'='*70}")
    print("  ALL DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
