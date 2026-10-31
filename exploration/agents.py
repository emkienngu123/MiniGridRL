from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Protocol

import numpy as np

from exploration.environment import DIRECTIONS


class Agent(Protocol):
    def reset(self, seed: int = 0) -> None: ...
    def act(self, observation: dict) -> int: ...


class RandomAgent:
    name = "random"
    training_seed = None

    def __init__(self, seed=11):
        self.policy_seed = int(seed)
        self.reset()

    def reset(self, seed=0):
        self.rng = np.random.default_rng(np.random.SeedSequence([self.policy_seed, int(seed)]))

    def act(self, observation):
        return int(self.rng.integers(4))


def turn_toward(heading: int, desired: int) -> int:
    diff = (desired - heading) % 4
    return 2 if diff == 0 else (1 if diff == 1 else 0)


class FrontierAgent:
    """Deterministic BFS over OBSERVED floor only; no simulator reference.

    Frontiers are observed floor cells adjacent to an unobserved cell. At a
    frontier, rotate to observe that cell before attempting further movement.
    Visible monsters and their immediate neighbors are treated as unsafe.
    """

    name = "frontier"
    training_seed = None

    def reset(self, seed=0):
        pass

    def act(self, observation):
        memory = observation["memory"]
        locations = np.argwhere(memory[6] > 0)
        if len(locations) != 1:
            raise ValueError("Frontier agent requires the observation-derived memory input")
        y, x = map(int, locations[0])
        start = (x, y)
        heading = int(np.argmax(observation["scalars"][:4]))
        h, w = memory.shape[1:]
        threats = {(int(mx), int(my)) for my, mx in np.argwhere(memory[5] > 0)}
        danger = set(threats)
        for mx, my in threats:
            danger.update((mx + dx, my + dy) for dx, dy in DIRECTIONS)
        walkable = {(int(px), int(py)) for py, px in np.argwhere(memory[2] > 0)} - danger
        walkable.add(start)

        def unknown_neighbors(point):
            px, py = point
            return [(d, (px + dx, py + dy)) for d, (dx, dy) in enumerate(DIRECTIONS)
                    if 0 <= px + dx < w and 0 <= py + dy < h
                    and memory[0, py + dy, px + dx] == 0]

        queue = deque([start])
        parent = {start: None}
        target = None
        while queue:
            point = queue.popleft()
            if unknown_neighbors(point):
                target = point
                break
            px, py = point
            for dx, dy in DIRECTIONS:
                nxt = px + dx, py + dy
                if nxt in walkable and nxt not in parent:
                    parent[nxt] = point
                    queue.append(nxt)
        if target == start:
            # Looking toward the closest unseen neighboring direction will reveal it.
            desired = min(unknown_neighbors(start), key=lambda item: ((item[0] - heading) % 4, item[0]))[0]
            if desired == heading:
                # MiniGrid's forward cell is normally visible; wait safely if a
                # transient occluder makes it unknown instead of walking blindly.
                return 3
            return turn_toward(heading, desired)
        if target is None:
            # No frontier currently reachable: prefer safe, least-visited neighbors.
            neighbors = [(x + dx, y + dy, d) for d, (dx, dy) in enumerate(DIRECTIONS)
                         if (x + dx, y + dy) in walkable]
            if not neighbors:
                return 3
            _, _, desired = min(neighbors, key=lambda p: (memory[4, p[1], p[0]], p[2]))
            return turn_toward(heading, desired)
        while parent[target] != start:
            target = parent[target]
        desired = DIRECTIONS.index((target[0] - x, target[1] - y))
        return turn_toward(heading, desired)


class PPOAgent:
    def __init__(self, checkpoint, device="cpu"):
        from exploration.policy import load_checkpoint

        self.model, self.metadata = load_checkpoint(Path(checkpoint), device=device)
        self.checkpoint = str(Path(checkpoint))
        self.name = self.metadata["variant"]
        self.training_seed = self.metadata["seed"]
        final_manifest = Path(checkpoint).with_name("final.json")
        self.completed_training_steps = self.metadata.get("completed_training_steps", self.metadata["training_steps"])
        if final_manifest.exists():
            import json
            completed = json.loads(final_manifest.read_text(encoding="utf-8"))
            if completed.get("seed") == self.training_seed and completed.get("variant") == self.name:
                self.completed_training_steps = max(self.completed_training_steps, completed["training_steps"])

    def reset(self, seed=0):
        pass  # Feed-forward policy; environment resets observation-derived memory.

    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)


def create_agent(name="frontier", checkpoint=None, seed=11, device="cpu"):
    if name == "random":
        return RandomAgent(seed)
    if name == "frontier":
        return FrontierAgent()
    if name == "ppo" and checkpoint:
        return PPOAgent(checkpoint, device)
    raise ValueError("Choose random, frontier, or ppo with --checkpoint")
