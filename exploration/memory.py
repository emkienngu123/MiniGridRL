from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from exploration.config import DEFAULTS
from exploration.environment import DIRECTIONS, ExplorationEnv
from minigrid.core.constants import OBJECT_TO_IDX

CANVAS = 55
CENTER = CANVAS // 2
# Memory channels: observed, wall, floor, POI, visit count, current monster, agent.
# Arrays use conventional [channel, y, x], while native MiniGrid uses [x, y, feature].
LOCAL_TYPES = [OBJECT_TO_IDX[x] for x in ("unseen", "empty", "wall", "floor", "ball")]
_VX, _VY = np.meshgrid(np.arange(7), np.arange(7), indexing="ij")
VIEW_OFFSETS = tuple(
    (dx * (6 - _VY) + DIRECTIONS[(direction + 1) % 4][0] * (_VX - 3),
     dy * (6 - _VY) + DIRECTIONS[(direction + 1) % 4][1] * (_VX - 3))
    for direction, (dx, dy) in enumerate(DIRECTIONS)
)


@dataclass
class Discovery:
    seen: int = 0
    visited: int = 0
    poi: int = 0


class ObservationMemory:
    """Pure observation consumer: no environment reference or ground-truth access."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.map = np.zeros((7, CANVAS, CANVAS), dtype=np.float32)
        self.position = (CENTER, CENTER)
        self.visits = np.zeros((CANVAS, CANVAS), dtype=np.int32)
        self.seen_floor: set[tuple[int, int]] = set()
        self.discovered_poi: set[tuple[int, int]] = set()
        self.heading = 0

    def update(self, obs: dict, action: int | None = None) -> Discovery:
        if action == 2 and obs["feedback"][0]:
            dx, dy = DIRECTIONS[self.heading]
            self.position = self.position[0] + dx, self.position[1] + dy
        self.heading = int(obs["direction"])
        x, y = self.position
        if not (0 <= x < CANVAS and 0 <= y < CANVAS):
            raise RuntimeError("Odometry exceeded the supported memory canvas")
        delta = Discovery(visited=int(self.visits[y, x] == 0))
        # A visit is an occupancy observation, including a stationary action.
        self.visits[y, x] += 1
        self.map[4, y, x] = min(self.visits[y, x], 10) / 10
        self.map[5:] = 0
        self.map[6, y, x] = 1
        types = obs["image"][:, :, 0]
        ox, oy = VIEW_OFFSETS[self.heading]
        wx, wy = x + ox, y + oy
        valid = ((types != OBJECT_TO_IDX["unseen"]) & (wx >= 0) & (wy >= 0)
                 & (wx < CANVAS) & (wy < CANVAS))
        wx, wy, types = wx[valid], wy[valid], types[valid]
        wall = types == OBJECT_TO_IDX["wall"]
        new_floor = ~wall & (self.map[2, wy, wx] == 0)
        new_poi = (types == OBJECT_TO_IDX["floor"]) & (self.map[3, wy, wx] == 0)
        delta.seen = int(np.count_nonzero(new_floor))
        delta.poi = int(np.count_nonzero(new_poi))
        self.seen_floor.update(zip(map(int, wx[new_floor]), map(int, wy[new_floor])))
        self.discovered_poi.update(zip(map(int, wx[new_poi]), map(int, wy[new_poi])))
        self.map[0, wy, wx] = 1
        self.map[1, wy, wx] = wall
        self.map[2, wy, wx] = ~wall
        self.map[3, wy[new_poi], wx[new_poi]] = 1
        monster = types == OBJECT_TO_IDX["ball"]
        self.map[5, wy[monster], wx[monster]] = 1
        return delta

    def encode(self, obs: dict, use_memory: bool = True) -> dict:
        types = obs["image"][:, :, 0].T
        local = np.stack([types == t for t in LOCAL_TYPES]).astype(np.float32)
        heading = np.eye(4, dtype=np.float32)[int(obs["direction"])]
        return {
            "local": local,
            "memory": self.map.copy() if use_memory else np.zeros_like(self.map),
            "scalars": np.concatenate([heading, obs["feedback"]]).astype(np.float32),
        }

    def render(self, scale=8):
        image = np.full((CANVAS, CANVAS, 3), (18, 25, 40), dtype=np.uint8)
        image[self.map[2] > 0] = (205, 214, 220)
        image[self.map[1] > 0] = (80, 94, 112)
        image[self.visits > 0] = (127, 183, 205)
        image[self.map[3] > 0] = (58, 193, 119)
        image[self.map[5] > 0] = (229, 78, 81)
        image[self.map[6] > 0] = (255, 190, 62)
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


class ExplorationWrapper(gym.Wrapper):
    def __init__(self, env: ExplorationEnv, variant="main", reward=None):
        super().__init__(env)
        if variant not in {"main", "no_memory", "no_novelty"}:
            raise ValueError("Unknown observation/reward variant")
        self.variant = variant
        self.weights = dict(DEFAULTS["reward"])
        self.weights.update(reward or {})
        self.memory = ObservationMemory()
        self.observation_space = spaces.Dict({
            "local": spaces.Box(0, 1, (5, 7, 7), np.float32),
            "memory": spaces.Box(0, 1, (7, CANVAS, CANVAS), np.float32),
            "scalars": spaces.Box(0, 1, (8,), np.float32),
        })
        self.wall_collisions = 0
        self.monster_collisions = 0
        self.loop_steps = 0
        self.repeated_steps = 0
        self.history = []
        self.coverage_history = []
        self.discovery_history = []
        self.total_reward = 0.0

    def reset(self, **kwargs):
        raw, _ = self.env.reset(**kwargs)
        start = perf_counter()
        self.memory.reset()
        self.memory.update(raw)
        self.wall_collisions = self.monster_collisions = self.loop_steps = self.repeated_steps = 0
        self.total_reward = 0.0
        self.history = [(*self.memory.position, self.memory.heading)]
        self.coverage_history = [len(self.memory.seen_floor)]
        self.discovery_history = [len(self.memory.discovered_poi)]
        encoded = self.memory.encode(raw, self.variant != "no_memory")
        self.preprocessing_seconds = perf_counter() - start
        return encoded, {}

    def step(self, action):
        raw, _, terminated, truncated, _ = self.env.step(action)
        start = perf_counter()
        delta = self.memory.update(raw, int(action))
        weights = self.weights
        terms = {
            "seen": weights["seen"] * delta.seen if self.variant != "no_novelty" else 0.0,
            "visited": weights["visited"] * delta.visited if self.variant != "no_novelty" else 0.0,
            "poi": weights["poi"] * delta.poi,
            "step": weights["step"],
            "wall": weights["wall"] * float(raw["feedback"][1]),
            "death": weights["death"] * float(raw["feedback"][2]),
        }
        self.wall_collisions += int(raw["feedback"][1])
        self.monster_collisions += int(raw["feedback"][2])
        state = (*self.memory.position, self.memory.heading)
        repeat = state in self.history[-16:]
        self.repeated_steps += int(repeat)
        self.loop_steps += int(repeat and delta.seen == 0 and delta.poi == 0)
        self.history.append(state)
        self.coverage_history.append(len(self.memory.seen_floor))
        self.discovery_history.append(len(self.memory.discovered_poi))
        reward = float(sum(terms.values()))
        self.total_reward += reward
        # These terms are observable. No coverage denominator or hidden state in info.
        encoded = self.memory.encode(raw, self.variant != "no_memory")
        self.preprocessing_seconds = perf_counter() - start
        return encoded, reward, terminated, truncated, {"reward_terms": terms}


def make_env(preset="easy", variant="main", reward=None, **kwargs):
    return ExplorationWrapper(ExplorationEnv(preset, **kwargs), variant, reward)
