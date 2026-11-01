"""Frozen-checkpoint memory diagnostics on validation maps only.

Outputs are separate from the submitted benchmark. Mask/translation interventions
are distribution shifts, useful for mechanism diagnosis, not new trained-agent
comparisons. No checkpoint or original result is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from exploration.agents import FrontierAgent
from exploration.environment import DIRECTIONS
from exploration.evaluation import episode_metrics, write_csv
from exploration.memory import make_env
from exploration.policy import load_checkpoint, make_model
from minigrid.core.constants import OBJECT_TO_IDX

ROOT = Path(__file__).resolve().parents[1]


def transform(observation, intervention):
    """Transform policy input only; reward/metrics retain their original history."""
    if intervention == "full":
        return observation
    memory = observation["memory"].copy()
    result = {**observation, "memory": memory}
    if intervention == "zero_memory":
        memory[:] = 0
    elif intervention == "no_visits":
        memory[4] = 0
    elif intervention == "no_agent_marker":
        memory[6] = 0
    elif intervention == "no_history":
        locations = np.argwhere(memory[6] > 0)
        if len(locations) != 1:
            raise ValueError("A full memory input with one agent marker is required")
        y, x = map(int, locations[0])
        heading = int(np.argmax(observation["scalars"][:4]))
        fx, fy = DIRECTIONS[heading]
        rx, ry = DIRECTIONS[(heading + 1) % 4]
        vy, vx = np.nonzero(observation["local"][0] == 0)
        wx, wy = x + fx * (6 - vy) + rx * (vx - 3), y + fy * (6 - vy) + ry * (vx - 3)
        valid = (wx >= 0) & (wy >= 0) & (wx < memory.shape[2]) & (wy < memory.shape[1])
        wx, wy, vx, vy = wx[valid], wy[valid], vx[valid], vy[valid]
        memory[:] = 0
        wall = observation["local"][2, vy, vx]
        memory[0, wy, wx] = 1
        memory[1, wy, wx] = wall
        memory[2, wy, wx] = 1 - wall
        memory[3, wy, wx] = observation["local"][3, vy, vx]
        memory[5, wy, wx] = observation["local"][4, vy, vx]
        memory[6, y, x] = 1
    elif intervention == "translate":
        active = np.argwhere(np.any(memory != 0, axis=0))
        if not len(active):
            return result
        lo, hi = active.min(axis=0), active.max(axis=0)
        for dy, dx in ((0, 4), (0, -4), (4, 0), (-4, 0)):
            if (lo[0] + dy >= 0 and hi[0] + dy < memory.shape[1]
                    and lo[1] + dx >= 0 and hi[1] + dx < memory.shape[2]):
                result["memory"] = np.roll(memory, (dy, dx), axis=(1, 2))
                break
        else:
            raise ValueError("No non-clipping translation available")
    elif intervention == "rotate90":
        result["memory"] = np.rot90(memory, axes=(1, 2)).copy()
        result["scalars"] = observation["scalars"].copy()
        heading = (int(np.argmax(result["scalars"][:4])) - 1) % 4
        result["scalars"][:4] = np.eye(4, dtype=np.float32)[heading]
    else:
        raise ValueError(intervention)
    return result


def tensors(model, observations):
    batch = {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}
    return model.policy.obs_to_tensor(batch)[0]


def probabilities(model, observations):
    with torch.inference_mode():
        return model.policy.get_distribution(tensors(model, observations)).distribution.probs.cpu().numpy()


class Audit:
    """Ground-truth checks are confined to this evaluator, never policy inputs."""

    def __init__(self):
        self.states = 0
        self.cells = 0
        self.headings = Counter()
        self.errors = Counter()
        self.known_fraction = []
        self.visit_saturation = []
        self.expected_observed = {}
        self.visible_cells = 0

    def check(self, env, origin):
        base, memory = env.unwrapped, env.memory
        mx, my = memory.position
        ax, ay = map(int, base.agent_pos)
        self.states += 1
        self.headings[int(base.agent_dir)] += 1
        self.errors["pose"] += int((mx, my) != (27 + ax - origin[0], 27 + ay - origin[1]))
        ys, xs = np.nonzero(memory.map[0])
        wx, wy = xs - 27 + origin[0], ys - 27 + origin[1]
        actual_floor = np.array([(int(x), int(y)) in base.floor_cells for x, y in zip(wx, wy)])
        self.errors["floor"] += int(np.count_nonzero(memory.map[2, ys, xs] != actual_floor))
        self.errors["wall"] += int(np.count_nonzero(memory.map[1, ys, xs] != ~actual_floor))
        self.cells += len(xs)
        pys, pxs = np.nonzero(memory.map[3])
        self.errors["poi"] += sum((int(x - 27 + origin[0]), int(y - 27 + origin[1])) not in base.poi_positions
                                  for x, y in zip(pxs, pys))
        self.errors["agent_marker"] += int(np.count_nonzero(memory.map[6]) != 1 or memory.map[6, my, mx] != 1)
        # Independent MiniGrid coordinate API checks current monster visibility.
        vx, vy = base.get_view_coords(*base.monster_pos)
        raw = base.gen_obs()["image"]
        if base.step_count == 0 or id(env) not in self.expected_observed:
            self.expected_observed[id(env)] = set()
        # MiniGrid's renderer defines the canonical world-space top-left corner.
        top_left = np.array(base.agent_pos) + base.dir_vec * 6 - base.right_vec * 3
        visible_world = set()
        for vx0, vy0 in np.argwhere(raw[:, :, 0] != OBJECT_TO_IDX["unseen"]):
            world = top_left - base.dir_vec * vy0 + base.right_vec * vx0
            wx0, wy0 = map(int, world)
            mx0, my0 = wx0 - origin[0] + 27, wy0 - origin[1] + 27
            if 0 <= mx0 < 55 and 0 <= my0 < 55:
                visible_world.add((wx0, wy0))
                self.errors["missing_current_visible"] += int(memory.map[0, my0, mx0] == 0)
                if int(raw[vx0, vy0, 0]) == OBJECT_TO_IDX["floor"]:
                    self.errors["missing_current_poi"] += int(memory.map[3, my0, mx0] == 0)
        self.visible_cells += len(visible_world)
        self.expected_observed[id(env)].update(visible_world)
        remembered_world = set(zip(map(int, wx), map(int, wy)))
        self.errors["forgotten_observations"] += len(self.expected_observed[id(env)] - remembered_world)
        visible = (0 <= vx < 7 and 0 <= vy < 7 and int(raw[vx, vy, 0]) == OBJECT_TO_IDX["ball"])
        actual_monsters = {(int(x - 27 + origin[0]), int(y - 27 + origin[1]))
                           for y, x in np.argwhere(memory.map[5] > 0)}
        expected = {base.monster_pos} if visible else set()
        self.errors["current_monster"] += int(actual_monsters != expected)
        self.known_fraction.append(float(np.mean(memory.map[0])))
        occupied = memory.visits > 0
        self.visit_saturation.append(float(np.mean(memory.visits[occupied] >= 10)))

    def summary(self):
        return {"checked_states": self.states, "checked_known_cells": self.cells,
                "checked_current_visible_cells": self.visible_cells,
                "headings": dict(self.headings), "errors": dict(self.errors),
                "mean_canvas_known_fraction": float(np.mean(self.known_fraction)),
                "mean_fraction_occupied_cells_saturated": float(np.mean(self.visit_saturation))}


class Reservoir:
    def __init__(self, limit=96):
        self.limit = limit
        self.observations = defaultdict(list)
        self.counts = Counter()
        self.rng = np.random.default_rng(6419)

    def add(self, key, observation):
        self.counts[key] += 1
        target = self.observations[key]
        index = int(self.rng.integers(self.counts[key]))
        if len(target) < self.limit:
            target.append({k: v.copy() for k, v in observation.items()})
        elif index < self.limit:
            target[index] = {k: v.copy() for k, v in observation.items()}


def rollout_batch(model, family, training_seed, cases, intervention, stochastic, audit, reservoir):
    episodes = []
    for preset, seed in cases:
        env = make_env(preset, variant=family)
        observation, _ = env.reset(seed=seed)
        episodes.append({"env": env, "obs": observation, "preset": preset, "seed": seed,
                         "origin": tuple(env.unwrapped.agent_pos),
                         "actions": np.zeros(4, dtype=np.int64), "entropy": 0.0, "margin": 0.0,
                         "rng": np.random.default_rng(np.random.SeedSequence([391, training_seed, seed]))})
    active = list(episodes)
    rows = []
    try:
        while active:
            observations = [transform(ep["obs"], intervention) for ep in active]
            probs = probabilities(model, observations)
            following = []
            for ep, distribution in zip(active, probs):
                env = ep["env"]
                if intervention == "full" and not stochastic and env.unwrapped.step_count % 32 == 0:
                    audit.check(env, ep["origin"])
                    if family == "main":
                        reservoir.add((training_seed, ep["preset"]), ep["obs"])
                if stochastic:
                    p = distribution.astype(np.float64)
                    action = int(ep["rng"].choice(4, p=p / p.sum()))
                else:
                    action = int(distribution.argmax())
                ep["actions"][action] += 1
                ep["entropy"] -= float(np.sum(distribution * np.log(distribution + 1e-9)))
                sorted_probability = np.sort(distribution)
                ep["margin"] += float(sorted_probability[-1] - sorted_probability[-2])
                ep["obs"], _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    row = episode_metrics(env, terminated, truncated)
                    rows.append({"family": family, "training_seed": training_seed,
                                 "intervention": intervention, "sampling": "stochastic" if stochastic else "greedy", **row,
                                 **{f"action_{name}_fraction": float(ep["actions"][index] / row["steps"])
                                    for index, name in enumerate(("left", "right", "forward", "wait"))},
                                 "mean_policy_entropy": ep["entropy"] / row["steps"],
                                 "mean_action_margin": ep["margin"] / row["steps"]})
                else:
                    following.append(ep)
            active = following
    finally:
        for ep in episodes:
            ep["env"].close()
    return rows


def frontier_controls(cases):
    rows = []
    for intervention in ("full", "no_history"):
        agent = FrontierAgent()
        for preset, seed in cases:
            env = make_env(preset)
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            try:
                while not (terminated or truncated):
                    action = agent.act(transform(observation, intervention))
                    observation, _, terminated, truncated, _ = env.step(action)
                rows.append({"family": "frontier", "training_seed": None, "intervention": intervention,
                             "sampling": "greedy", **episode_metrics(env, terminated, truncated),
                             **{f"action_{name}_fraction": None for name in ("left", "right", "forward", "wait")},
                             "mean_policy_entropy": None, "mean_action_margin": None})
            finally:
                env.close()
    return rows


def audit_mapping(cases):
    """Independent completeness/retention checks along frontier trajectories."""
    audit = Audit()
    for preset, seed in cases:
        env = make_env(preset)
        observation, _ = env.reset(seed=seed)
        origin = tuple(env.unwrapped.agent_pos)
        agent = FrontierAgent()
        agent.reset(seed)
        try:
            while True:
                if env.unwrapped.step_count % 16 == 0:
                    audit.check(env, origin)
                observation, _, terminated, truncated, _ = env.step(agent.act(observation))
                if terminated or truncated:
                    break
        finally:
            env.close()
    result = audit.summary()
    if any(result["errors"].values()):
        raise AssertionError(f"Mapping audit failed: {result}")
    return result


def probe(model, observations):
    variants = (("full", "zero_memory", "no_history", "no_visits", "no_agent_marker", "translate", "rotate90")
                if np.any(observations[0]["memory"][6]) else ("full", "translate", "rotate90"))
    probs, values = {}, {}
    norm_local, norm_memory, actor_saturation, critic_saturation = [], [], [], []
    for start in range(0, len(observations), 64):
        chunk = observations[start:start + 64]
        for variant in variants:
            obs = [transform(o, variant) for o in chunk]
            tensor = tensors(model, obs)
            with torch.inference_mode():
                features = model.policy.extract_features(tensor)
                pi, vf = model.policy.mlp_extractor(features)
                p = model.policy.action_net(pi).softmax(dim=1).cpu().numpy()
                value = model.policy.value_net(vf).flatten().cpu().numpy()
                if variant == "full":
                    norm_local.extend(features[:, :64].norm(dim=1).cpu().tolist())
                    norm_memory.extend(features[:, 64:192].norm(dim=1).cpu().tolist())
                    actor_saturation.extend((model.policy.mlp_extractor.policy_net[0](features).abs() > 3).float().mean(dim=1).cpu().tolist())
                    critic_saturation.extend((model.policy.mlp_extractor.value_net[0](features).abs() > 3).float().mean(dim=1).cpu().tolist())
            probs.setdefault(variant, []).append(p)
            values.setdefault(variant, []).append(value)
    probs = {k: np.concatenate(v) for k, v in probs.items()}
    values = {k: np.concatenate(v) for k, v in values.items()}
    original = probs["full"]
    result = {"states": len(observations), "local_feature_l2": float(np.mean(norm_local)),
              "memory_feature_l2": float(np.mean(norm_memory)),
              "actor_tanh_saturated_fraction": float(np.mean(actor_saturation)),
              "critic_tanh_saturated_fraction": float(np.mean(critic_saturation)), "interventions": {}}
    for variant in variants[1:]:
        other = probs[variant]
        result["interventions"][variant] = {
            "greedy_action_flip_fraction": float(np.mean(original.argmax(axis=1) != other.argmax(axis=1))),
            "mean_total_variation": float(np.mean(np.abs(original - other).sum(axis=1) / 2)),
            "mean_kl_full_to_intervened": float(np.mean(np.sum(original * np.log((original + 1e-9) / (other + 1e-9)), axis=1))),
            "value_mean_absolute_change": float(np.mean(np.abs(values["full"] - values[variant]))),
        }
    return result


def weight_audit(model, config):
    original_config = dict(config, device="cpu")
    env = make_env("easy", variant=config["variant"])
    initial = make_model(env, original_config)
    differences = {}
    for name in ("memory", "local"):
        before = getattr(initial.policy.features_extractor, name).state_dict()
        after = getattr(model.policy.features_extractor, name).state_dict()
        delta = torch.cat([(after[k].cpu() - before[k].cpu()).flatten() for k in before])
        differences[name] = {"parameters": sum(v.numel() for v in before.values()),
                             "changed_parameters": int(torch.count_nonzero(delta)),
                             "weight_delta_l2": float(delta.norm())}
    differences["total_parameters"] = sum(p.numel() for p in model.policy.parameters())
    for name in ("policy_net", "value_net"):
        before = getattr(initial.policy.mlp_extractor, name)[0].weight[:, 64:192]
        after = getattr(model.policy.mlp_extractor, name)[0].weight[:, 64:192]
        differences[f"{name}_memory_columns_delta_l2"] = float((after.cpu() - before.cpu()).detach().norm())
    env.close()
    return differences


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = f"{row['family']}/{row['intervention']}/{row['sampling']}/{row['preset']}"
        grouped[key].append(row)
    return {key: {"episodes": len(group), **{
        metric: float(np.mean([r[metric] for r in group]))
        for metric in ("revealed_coverage", "coverage_auc", "poi_fraction", "survived", "loop_rate", "visited_coverage")},
        **{metric: float(np.mean([r[metric] for r in group])) for metric in
           ("action_wait_fraction", "action_forward_fraction", "mean_policy_entropy", "mean_action_margin")
           if all(r[metric] is not None for r in group)}}
        for key, group in sorted(grouped.items())}


def write_report(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def cohort(family, mode, sampling="greedy"):
        groups = [result["rollouts"][f"{family}/{mode}/{sampling}/{p}"] for p in ("easy", "medium", "hard")]
        return {"episodes": sum(g["episodes"] for g in groups), **{
            key: float(np.mean([g[key] for g in groups]))
            for key in ("revealed_coverage", "loop_rate", "survived")}}

    cohorts = {}
    for family, mode, sampling in (
        ("main", "full", "greedy"), ("main", "full", "stochastic"),
        ("main", "zero_memory", "greedy"), ("main", "no_history", "greedy"),
        ("main", "no_visits", "greedy"), ("no_memory", "full", "greedy"),
        ("no_memory", "full", "stochastic"), ("frontier", "full", "greedy"),
        ("frontier", "no_history", "greedy"),
    ):
        cohorts[f"{family}/{mode}/{sampling}"] = cohort(family, mode, sampling)

    probe_summary = {}
    for family in ("main", "no_memory"):
        groups = [g for key, g in result["policy_probes"].items()
                  if key.startswith("no_memory") == (family == "no_memory")]
        weights = [g["states"] for g in groups]
        probe_summary[family] = {"states": sum(weights), "interventions": {}}
        for mode in groups[0]["interventions"]:
            probe_summary[family]["interventions"][mode] = {
                key: float(np.average([g["interventions"][mode][key] for g in groups], weights=weights))
                for key in groups[0]["interventions"][mode]}
        for key in ("local_feature_l2", "memory_feature_l2", "actor_tanh_saturated_fraction", "critic_tanh_saturated_fraction"):
            probe_summary[family][key] = float(np.average([g[key] for g in groups], weights=weights))

    history = {}
    for family in ("main", "no_memory"):
        records = []
        for seed in result["training_seeds"]:
            folder = ROOT / f"runs/{family}-{seed}"
            with (folder / "validation.csv").open(encoding="utf-8", newline="") as stream:
                values = list(csv.DictReader(stream))
            history[f"{family}-{seed}"] = {
                "best_auc": max(float(r["coverage_auc"]) for r in values),
                "final_auc": float(values[-1]["coverage_auc"])}
            records.extend((int(r["step"]), float(r["coverage_auc"])) for r in values if int(r["step"]) % 250000 == 0)
        history[family] = {"steps": sorted({r[0] for r in records})}
        history[family]["mean_auc"] = [float(np.mean([r[1] for r in records if r[0] == step])) for step in history[family]["steps"]]
        history[family]["std_auc"] = [float(np.std([r[1] for r in records if r[0] == step], ddof=1)) for step in history[family]["steps"]]

    combined = {"cohorts": cohorts, "probes": probe_summary, "validation_history": history}
    (output / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for family, color in (("main", "#377ad3"), ("no_memory", "#a368b5")):
        h = history[family]
        x = np.array(h["steps"]) / 1e6
        y, std = np.array(h["mean_auc"]), np.array(h["std_auc"])
        axes[0].plot(x, y, label=family, color=color)
        axes[0].fill_between(x, y - std, y + std, color=color, alpha=0.15)
    axes[0].set(xlabel="Training transitions (millions)", ylabel="Validation coverage AUC", ylim=(0, 1),
                title="Mean +/- SD across training runs")
    axes[0].legend(frameon=False)
    labels = ["full / greedy", "full / sampled", "zero map", "current view only", "no visit counts"]
    keys = ["main/full/greedy", "main/full/stochastic", "main/zero_memory/greedy",
            "main/no_history/greedy", "main/no_visits/greedy"]
    axes[1].barh(labels[::-1], [cohorts[k]["revealed_coverage"] * 100 for k in keys][::-1], color="#377ad3")
    axes[1].set(xlabel="Mean validation coverage (%)", xlim=(0, 100), title="Frozen main policies: input/action interventions")
    fig.tight_layout()
    fig.savefig(output / "diagnostics.png", dpi=180)
    plt.close(fig)

    audit = result["audit"]
    supplementary = output / "mapping-audit.json"
    extra_audit = json.loads(supplementary.read_text(encoding="utf-8")) if supplementary.exists() else None
    main = cohorts["main/full/greedy"]
    stochastic = cohorts["main/full/stochastic"]
    zero = cohorts["main/zero_memory/greedy"]
    current = cohorts["main/no_history/greedy"]
    pro = probe_summary["main"]
    pct = lambda value: f"{value * 100:.1f}%"
    text = ["# Memory investigation", "",
            ("This is a post-benchmark investigation using frozen checkpoints and validation seeds "
             f"{result['map_seeds'][0]}-{result['map_seeds'][-1]}. It does not change the submitted benchmark, models, or report."), "",
            "## Confirmed findings", "",
            f"- Mapping audit: {audit['checked_states']:,} states and {audit['checked_known_cells']:,} known-cell checks across all four headings. Errors: {audit['errors']}.",
            *([f"- A supplementary frontier-trajectory audit checks {extra_audit['checked_states']:,} states and {extra_audit['checked_current_visible_cells']:,} currently visible cells. There are no missing visible cells/POIs or forgotten sampled observations. Full results: `mapping-audit.json`."] if extra_audit else []),
            "- The no_memory variant still maintains observation history for reward calculation. The original ablation removes the map from policy inputs, not history from the entire system.",
            f"- The trained memory policies depend on the map: zeroing it changes greedy actions in {pct(pro['interventions']['zero_memory']['greedy_action_flip_fraction'])} of {pro['states']} sampled states. Mean coverage falls from {pct(main['revealed_coverage'])} to {pct(zero['revealed_coverage'])}. Keeping only current-view information gives {pct(current['revealed_coverage'])}.",
            f"- Sampling actions improves the main policies' mean coverage from {pct(main['revealed_coverage'])} to {pct(stochastic['revealed_coverage'])}, with loop rate falling from {pct(main['loop_rate'])} to {pct(stochastic['loop_rate'])}. The memory-disabled policies also improve under sampling, so sampling does not explain the entire gap.",
            f"- Removing the agent-position marker changes {pct(pro['interventions']['no_agent_marker']['greedy_action_flip_fraction'])} of greedy actions; average probability total variation is {pro['interventions']['no_agent_marker']['mean_total_variation']:.4f}. Pose can be redundant with other channels, so this is weak marker reliance, not proof of no localization.",
            f"- A non-clipping four-cell translation of every map channel changes {pct(pro['interventions']['translate']['greedy_action_flip_fraction'])} of main-policy decisions. Joint map/heading rotation changes {pct(pro['interventions']['rotate90']['greedy_action_flip_fraction'])}, versus {pct(probe_summary['no_memory']['interventions']['rotate90']['greedy_action_flip_fraction'])} for memory-disabled policies on the same local observations.",
            "- The map encoder has 459,008 parameters out of 649,925 total (70.6%). Every one of those parameters remains exactly at initialization in all three no_memory checkpoints. Their memory-to-actor and memory-to-critic columns also remain unchanged. Thus the nominally identical architecture has substantially less active learning capacity in the no_memory condition.",
            f"- Average known canvas fraction is {pct(audit['mean_canvas_known_fraction'])}. Actor tanh saturation above |preactivation|=3 affects only {pct(pro['actor_tanh_saturated_fraction'])} of units, so these probes do not support a simple activation-explosion explanation.", "",
            "## Frozen-checkpoint rollout results", "",
            "Means pool Easy/Medium/Hard equally. Learned-policy rows each use three training seeds; these are validation diagnostics, not final test scores.", "",
            "| Family | Policy input | Action choice | Episodes | Coverage | Loop rate | Survival |",
            "|---|---|---|---:|---:|---:|---:|"]
    for key, values in cohorts.items():
        family, mode, choice = key.split("/")
        text.append(f"| {family} | {mode} | {choice} | {values['episodes']} | {pct(values['revealed_coverage'])} | {pct(values['loop_rate'])} | {pct(values['survived'])} |")
    text += ["", "## Interpretation and limits", "",
             "The evidence supports a representation/optimization problem plus greedy-policy looping, rather than corrupted or wholly ignored memory. The local view is agent-relative, while the large map is anchored at the episode start and aligned to world directions. A three-stage strided CNN followed by flattening must learn localization, frame alignment, history interpretation, and navigation from PPO rewards. No explicit agent-centered crop or frontier-goal representation is supplied.", "",
             "This is a supported mechanism hypothesis, not an isolated causal proof. Masked and translated maps are out-of-distribution interventions. Agent position is also partially recoverable from visit patterns. The stochastic comparison has one action RNG stream per model/map; more sampling seeds would strengthen it. Frontier without history is a positive control for its map-dependent algorithm, not a proposed competitive stateless baseline.", "",
             "The main model's weaker results are not explained solely by the resumed seed-33 run: the original held-out comparison still favors no_memory when comparing only seeds 11 and 22. Main validation AUC also regresses substantially late in all three runs, whereas no_memory is more stable; simply extending this training setup is not established as a solution.", "",
             "## Next controlled experiment", "",
             "1. Encode a smaller agent-centered, heading-aligned memory patch, including visit counts and known/unknown cells. This removes the need to infer the agent's location from a single pixel on a world-aligned 55x55 map.",
             "2. Use a smaller map encoder and distinguish input-information effects from active-parameter-count effects with a capacity-matched local-only control.",
             "3. Evaluate both greedy and sampled actions under a declared validation protocol. Select the mode on validation and reserve a fresh held-out set for the revised system.",
             "4. Consider warm-starting from the successful local policy, initially zeroing only the map-to-head connection weights so adding memory preserves initial behavior while allowing new map connections to learn.", "",
             "No revised policy has been trained by this investigation. These changes require matched-budget, multi-seed retraining before claiming an improvement.", "",
             "## Reproduction and sources", "",
             "```powershell", "..\\.venv\\Scripts\\python.exe scripts/investigate_memory.py --count 10",
             "..\\.venv\\Scripts\\python.exe scripts/investigate_memory.py --count 10 --mapping-only", "```", "",
             "- Raw evidence: `results/memory-investigation/diagnostics.json` and `episodes.csv`; aggregates: `summary.json`.",
             f"- Original benchmark summary SHA-256: `{result['baseline_results_sha256']}`; unchanged: `{result['baseline_results_unchanged']}`.",
             "- Source: `exploration/memory.py`, `policy.py`, `training.py`; diagnostic transformations have independent visibility/geometry tests.",
             "- SB3 documents shared PPO feature extractors: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html . Shared extraction is confirmed in the loaded models; it is not proven to be the cause here.",
             "- Architectural context: Active Neural SLAM uses modular mapping, global/local policies, and analytic planning: https://arxiv.org/abs/2004.05155 . This motivates a structured map interface, not a claim that it will automatically improve this task.", ""]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--output", default="results/memory-investigation")
    parser.add_argument("--mapping-only", action="store_true", help="Run only the supplementary completeness/retention audit")
    args = parser.parse_args()
    if not 1 <= args.count <= 50:
        raise ValueError("Diagnostics use 1-50 validation seeds")
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = ROOT / "results/full/summary.json"
    baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    cases = [(p, s) for p in ("easy", "medium", "hard") for s in range(10000, 10000 + args.count)]
    if args.mapping_only:
        result = audit_mapping(cases)
        (output / "mapping-audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if (output / "diagnostics.json").exists():
            write_report(json.loads((output / "diagnostics.json").read_text(encoding="utf-8")), output)
        print(json.dumps(result, indent=2))
        return
    audit, reservoir = Audit(), Reservoir()
    rows, probes, weights = [], {}, {}
    started = time.perf_counter()
    for family in ("main", "no_memory"):
        for seed in args.seeds:
            model, metadata = load_checkpoint(ROOT / f"runs/{family}-{seed}/best.zip", "cpu")
            weights[f"{family}-{seed}"] = weight_audit(model, metadata["config"])
            conditions = [("full", False), ("full", True)]
            if family == "main":
                conditions += [("zero_memory", False), ("no_history", False), ("no_visits", False)]
            for intervention, stochastic in conditions:
                new_rows = rollout_batch(model, family, seed, cases, intervention, stochastic, audit, reservoir)
                rows.extend(new_rows)
                write_csv(output / "episodes.csv", rows)
                print(f"{family}-{seed} {intervention} stochastic={stochastic}: "
                      f"coverage={np.mean([r['revealed_coverage'] for r in new_rows]):.3f} "
                      f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
            if family == "main":
                for preset in ("easy", "medium", "hard"):
                    probes[f"{seed}/{preset}"] = probe(model, reservoir.observations[seed, preset])
            else:
                for preset in ("easy", "medium", "hard"):
                    observations = [transform(obs, "zero_memory") for obs in reservoir.observations[seed, preset]]
                    probes[f"no_memory-{seed}/{preset}"] = probe(model, observations)
    rows.extend(frontier_controls(cases))
    write_csv(output / "episodes.csv", rows)
    result = {"scope": "post-benchmark diagnostics; validation maps only; no parameter updates",
              "map_seeds": list(range(10000, 10000 + args.count)), "training_seeds": args.seeds,
              "baseline_results_sha256": baseline_hash,
              "baseline_results_unchanged": hashlib.sha256(baseline_path.read_bytes()).hexdigest() == baseline_hash,
              "audit": audit.summary(), "weights": weights, "policy_probes": probes,
              "rollouts": aggregate(rows), "elapsed_seconds": time.perf_counter() - started,
              "caveat": "Counterfactual masking/translation inputs are out of the trained distribution; rollout changes are diagnostic, not a replacement benchmark."}
    (output / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, output)
    print(json.dumps({"audit": result["audit"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
