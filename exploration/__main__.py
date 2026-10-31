from __future__ import annotations

import argparse
import json
import multiprocessing
import sys

from exploration.config import PRESETS, load_config, seed_values


def parser():
    root = argparse.ArgumentParser(description="MiniGrid exploration: train, evaluate, inspect, and replay")
    commands = root.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train PPO using only the training seed partition")
    training.add_argument("--config", default=None)
    for name, typ in (("seed", int), ("timesteps", int), ("n-envs", int), ("eval-every", int)):
        training.add_argument(f"--{name}", type=typ)
    training.add_argument("--variant", choices=["main", "no_memory", "no_novelty"])
    training.add_argument("--device", choices=["auto", "cpu", "cuda"])
    training.add_argument("--vector-backend", choices=["dummy", "subproc"])
    training.add_argument("--output")
    training.add_argument("--resume", help="Checkpoint to continue to the total --timesteps target")
    evaluation = commands.add_parser("evaluate", help="Evaluate baselines and/or saved policies")
    evaluation.add_argument("--config")
    evaluation.add_argument("--agents", nargs="+", choices=["random", "frontier", "ppo"], default=["random", "frontier"])
    evaluation.add_argument("--checkpoints", nargs="*", default=[])
    evaluation.add_argument("--presets", nargs="+", choices=list(PRESETS), default=list(PRESETS))
    evaluation.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    evaluation.add_argument("--count", type=int)
    evaluation.add_argument("--policy-seeds", type=int, nargs="+", default=[11, 22, 33])
    evaluation.add_argument("--output", default="results/evaluation")
    replay = commands.add_parser("replay", help="Export an episode, action trace, and video")
    replay.add_argument("--agent", choices=["random", "frontier", "ppo"], default="frontier")
    replay.add_argument("--checkpoint")
    replay.add_argument("--seed", type=int, default=10000)
    replay.add_argument("--preset", choices=list(PRESETS), default="easy")
    replay.add_argument("--output", default="results/replay")
    ui = commands.add_parser("ui", help="Launch the interactive Python inspector")
    ui.add_argument("--checkpoint")
    smoke = commands.add_parser("smoke", help="Run a short headless episode; optionally verify a checkpoint")
    smoke.add_argument("--checkpoint")
    smoke.add_argument("--steps", type=int, default=64)
    commands.add_parser("matrix", help="Print commands for all nine full training runs")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "train":
        from exploration.training import train
        config = load_config(args.config)
        for key in ("seed", "timesteps", "n_envs", "eval_every", "variant", "device", "vector_backend", "output"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        if args.output is None and (args.seed is not None or args.variant is not None):
            config["output"] = f"runs/{config['variant']}-{config['seed']}"
        train(config, resume=args.resume)
    elif args.command == "evaluate":
        import torch

        from exploration.agents import create_agent
        from exploration.evaluation import evaluate
        torch.set_num_threads(1)
        config = load_config(args.config)
        agents = []
        if "random" in args.agents:
            agents.extend(create_agent("random", seed=s) for s in args.policy_seeds)
        if "frontier" in args.agents:
            agents.append(create_agent("frontier"))
        if "ppo" in args.agents:
            if not args.checkpoints:
                raise ValueError("--agents ppo requires --checkpoints")
            agents.extend(create_agent("ppo", checkpoint=p) for p in args.checkpoints)
        evaluate(agents, args.presets, seed_values(config, args.split, args.count), args.output, split=args.split)
    elif args.command == "replay":
        import torch

        from exploration.replay import replay
        torch.set_num_threads(1)
        replay(args.agent, args.checkpoint, args.seed, args.preset, args.output)
    elif args.command == "ui":
        from exploration.ui import launch
        launch(args.checkpoint)
    elif args.command == "smoke":
        from exploration.agents import create_agent
        from exploration.evaluation import run_episode
        if args.steps < 1:
            raise ValueError("--steps must be positive")
        if getattr(sys, "frozen", False):
            # Validate the UI Python/native modules without requiring a display
            # or starting Tcl in a restricted/headless build runner.
            __import__("exploration.ui")
        if args.checkpoint:
            import torch
            torch.set_num_threads(1)
        agent = create_agent("ppo" if args.checkpoint else "frontier", args.checkpoint)
        variant = getattr(agent, "metadata", {}).get("variant", "main")
        metrics, _, frames, _ = run_episode(agent, "easy", 123, variant=variant,
                                            env_kwargs={"max_steps": args.steps}, capture=True)
        assert frames and frames[0].ndim == 3
        print(json.dumps({"smoke": "passed", "checkpoint_loaded": bool(args.checkpoint),
                          "frozen": bool(getattr(sys, "frozen", False)),
                          "ui_import_verified": bool(getattr(sys, "frozen", False)),
                          "inference_device": str(agent.model.device) if hasattr(agent, "model") else "cpu",
                          **metrics}, indent=2))
    elif args.command == "matrix":
        for variant in ("main", "no_memory", "no_novelty"):
            for seed in (11, 22, 33):
                print(f"python -m exploration train --config configs/main.yaml --variant {variant} --seed {seed} --output runs/{variant}-{seed}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
