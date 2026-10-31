from __future__ import annotations

from collections import deque

import numpy as np
from gymnasium import spaces

from exploration.config import PRESETS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Floor, Wall
from minigrid.minigrid_env import MiniGridEnv

DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


class Monster(Ball):
    def __init__(self):
        super().__init__("red")

    def can_overlap(self):
        return True


def distances(floor: set[tuple[int, int]], start: tuple[int, int]) -> dict:
    result = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            nxt = x + dx, y + dy
            if nxt in floor and nxt not in result:
                result[nxt] = result[x, y] + 1
                queue.append(nxt)
    return result


class ExplorationEnv(MiniGridEnv):
    """MiniGrid transitions/visibility with seeded procedural maps and a patrol.

    Public observations contain only MiniGrid's view and proprioceptive feedback.
    Ground truth attributes are for simulation/evaluation, never agent inputs.
    """

    def __init__(self, preset="easy", *, size=None, poi_count=None, max_steps=None,
                 monster_interval=None, render_mode="rgb_array"):
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}")
        settings = dict(PRESETS[preset])
        overrides = {"size": size, "poi_count": poi_count, "max_steps": max_steps, "monster_interval": monster_interval}
        settings.update({k: v for k, v in overrides.items() if v is not None})
        self.preset = preset
        size = int(settings["size"])
        if size < 9 or size > 27 or size % 2 == 0:
            raise ValueError("Grid size must be odd and between 9 and 27")
        self.poi_count = int(settings["poi_count"])
        self.monster_interval = int(settings["monster_interval"])
        if self.poi_count < 1 or self.poi_count > size or self.monster_interval < 1 or settings["max_steps"] < 1:
            raise ValueError("Invalid POI count, monster interval, or episode limit")
        self.feedback = np.zeros(4, dtype=np.float32)
        self.ended = False
        super().__init__(
            MissionSpace(mission_func=lambda: "explore, discover green tiles, and avoid the red monster"),
            grid_size=size, max_steps=int(settings["max_steps"]), agent_view_size=7,
            see_through_walls=False, render_mode=render_mode,
        )
        self.action_space = spaces.Discrete(4)
        self.observation_space.spaces["feedback"] = spaces.Box(0.0, 1.0, (4,), np.float32)
        self.reward_range = (-float("inf"), float("inf"))

    def _gen_grid(self, width, height):
        # Independent streams: policy-dependent simulation cannot consume map RNG.
        episode_seed = int(self.np_random.integers(0, 2**63))
        map_seed, monster_seed = np.random.SeedSequence(episode_seed).spawn(2)
        rng = np.random.default_rng(map_seed)
        self.monster_rng = np.random.default_rng(monster_seed)
        self.grid = Grid(width, height)
        for x in range(width):
            for y in range(height):
                self.grid.set(x, y, Wall())
        # A depth-first maze guarantees connected corridors and dead-end branches.
        floor = {(1, 1)}
        stack = [(1, 1)]
        while stack:
            x, y = stack[-1]
            candidates = [(x + 2 * dx, y + 2 * dy) for dx, dy in DIRECTIONS
                          if 0 < x + 2 * dx < width - 1 and 0 < y + 2 * dy < height - 1
                          and (x + 2 * dx, y + 2 * dy) not in floor]
            if not candidates:
                stack.pop()
                continue
            nx, ny = candidates[int(rng.integers(len(candidates)))]
            floor.update({(nx, ny), ((x + nx) // 2, (y + ny) // 2)})
            stack.append((nx, ny))
        # Carve rooms into the connected maze and open a few additional cycles.
        for _ in range(max(2, width // 4)):
            rw, rh = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            rx, ry = int(rng.integers(1, width - rw)), int(rng.integers(1, height - rh))
            floor.update((x, y) for x in range(rx, rx + rw) for y in range(ry, ry + rh))
        for x in range(2, width - 2):
            for y in range(2, height - 2):
                if ((x, y) not in floor and rng.random() < 0.025 and
                    (((x - 1, y) in floor and (x + 1, y) in floor) or ((x, y - 1) in floor and (x, y + 1) in floor))):
                    floor.add((x, y))
        for point in floor:
            self.grid.set(*point, None)
        ordered = sorted(floor)
        self.agent_pos = ordered[int(rng.integers(len(ordered)))]
        self.agent_dir = int(rng.integers(4))
        reach = distances(floor, self.agent_pos)
        if len(reach) != len(floor):
            raise RuntimeError("Generator produced disconnected terrain")
        far = sorted(p for p, distance in reach.items() if distance >= 6)
        if not far:
            raise RuntimeError("No valid monster spawn")
        self.monster_pos = far[int(rng.integers(len(far)))]
        candidates = [p for p in ordered if p not in {self.agent_pos, self.monster_pos}]
        indices = rng.choice(len(candidates), self.poi_count, replace=False)
        self.poi_positions = {candidates[int(i)] for i in indices}
        for point in self.poi_positions:
            self.grid.set(*point, Floor("green"))
        self.floor_cells = floor
        self.monster = Monster()
        self.monster_under = self.grid.get(*self.monster_pos)
        self.grid.set(*self.monster_pos, self.monster)
        self.monster_previous = None
        self.feedback = np.array([0, 0, 0, 1], dtype=np.float32)
        self.ended = False
        self.mission = "explore, discover green tiles, and avoid the red monster"

    def gen_obs(self):
        obs = super().gen_obs()
        obs["feedback"] = self.feedback.copy()
        return obs

    def _move_monster(self):
        x, y = self.monster_pos
        legal = [(x + dx, y + dy) for dx, dy in DIRECTIONS
                 if (x + dx, y + dy) in self.floor_cells]
        preferred = [p for p in legal if p != self.monster_previous] or legal
        target = preferred[int(self.monster_rng.integers(len(preferred)))]
        self.grid.set(*self.monster_pos, self.monster_under)
        self.monster_previous = self.monster_pos
        self.monster_pos = target
        self.monster_under = self.grid.get(*target)
        self.grid.set(*target, self.monster)

    def step(self, action):
        if self.ended:
            raise RuntimeError("Episode finished; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected 0..3")
        action = int(action)
        old_pos = tuple(self.agent_pos)
        fwd = tuple(self.front_pos)
        wall = action == 2 and fwd not in self.floor_cells
        # MiniGrid executes turn/forward; its unused 'done' action is our wait.
        obs, _, _, truncated, _ = super().step(self.actions.done if action == 3 else action)
        death = tuple(self.agent_pos) == self.monster_pos
        if not death and self.step_count % self.monster_interval == 0:
            self._move_monster()
            death = tuple(self.agent_pos) == self.monster_pos
            obs = self.gen_obs()
        self.feedback = np.array([
            tuple(self.agent_pos) != old_pos, wall, death,
            max(0, 1 - self.step_count / self.max_steps),
        ], dtype=np.float32)
        self.ended = bool(death or truncated)
        obs["feedback"] = self.feedback.copy()
        return obs, 0.0, bool(death), bool(truncated and not death), {}
