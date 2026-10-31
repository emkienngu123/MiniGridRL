from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from exploration.environment import ExplorationEnv, Monster, distances
from exploration.memory import ExplorationWrapper, make_env
from minigrid.core.grid import Grid
from minigrid.core.world_object import Floor, Wall


class FixtureEnv(ExplorationEnv):
    """Small auditable layout. P=(2,2), POI=(3,2), monster=(7,7)."""

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.floor_cells = {(x, y) for x in range(1, width - 1) for y in range(1, height - 1)}
        self.agent_pos, self.agent_dir = (2, 2), 0
        self.poi_positions = {(3, 2)}
        self.poi_count = 1
        self.grid.set(3, 2, Floor("green"))
        self.monster_pos = (7, 7)
        self.monster_previous = None
        self.monster_under = None
        self.monster = Monster()
        self.grid.set(*self.monster_pos, self.monster)


@pytest.fixture
def fixed():
    env = ExplorationWrapper(FixtureEnv(size=9, poi_count=1, monster_interval=10000))
    env.reset(seed=7)
    yield env
    env.close()


def place_monster(base, position):
    base.grid.set(*base.monster_pos, base.monster_under)
    base.monster_pos = position
    base.monster_under = base.grid.get(*position)
    base.grid.set(*position, base.monster)


def test_gymnasium_contract():
    env = make_env("easy")
    check_env(env, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("preset", ["easy", "medium", "hard"])
def test_reproducible_complete_trajectory(preset):
    a, b = make_env(preset), make_env(preset)
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    for k in oa:
        np.testing.assert_array_equal(oa[k], ob[k])
    for action in np.random.default_rng(9).integers(4, size=180):
        ra, rb = a.step(int(action)), b.step(int(action))
        for k in ra[0]:
            np.testing.assert_array_equal(ra[0][k], rb[0][k])
        assert ra[1:] == rb[1:]
        assert a.unwrapped.monster_pos == b.unwrapped.monster_pos
        if ra[2] or ra[3]:
            break
    a.close()
    b.close()


@pytest.mark.parametrize("preset", ["easy", "medium", "hard"])
def test_generated_maps_connected_and_spawns_safe(preset):
    env = ExplorationEnv(preset)
    fingerprints = set()
    for seed in range(30):
        env.reset(seed=seed)
        reach = distances(env.floor_cells, tuple(env.agent_pos))
        assert len(reach) == len(env.floor_cells)
        assert reach[env.monster_pos] >= 6
        assert env.poi_positions <= set(reach)
        assert len(env.poi_positions) == env.poi_count
        assert env.agent_pos not in env.poi_positions
        assert all(0 < x < env.width - 1 and 0 < y < env.height - 1 for x, y in reach)
        fingerprints.add(env.grid.encode().tobytes())
    assert len(fingerprints) == 30
    env.close()


def test_native_turn_move_wait_and_odometry(fixed):
    start = fixed.memory.position
    fixed.step(2)
    assert fixed.unwrapped.agent_pos == (3, 2)
    assert fixed.memory.position == (start[0] + 1, start[1])
    fixed.step(1)
    fixed.step(2)
    assert fixed.unwrapped.agent_pos == (3, 3)
    assert fixed.memory.position == (start[0] + 1, start[1] + 1)
    fixed.step(0)
    before = fixed.unwrapped.agent_pos
    fixed.step(3)
    assert fixed.unwrapped.agent_pos == before
    assert fixed.unwrapped.agent_dir == 0


def test_wall_contact_does_not_move_memory(fixed):
    base = fixed.unwrapped
    base.grid.set(3, 2, Wall())
    base.floor_cells.remove((3, 2))
    before = fixed.memory.position
    _, _, died, _, info = fixed.step(2)
    assert not died
    assert fixed.memory.position == before
    assert fixed.wall_collisions == 1
    assert info["reward_terms"]["wall"] == -0.02


def test_monster_respects_walls_and_interval():
    env = ExplorationEnv("medium", max_steps=120)
    env.reset(seed=22)
    for step in range(1, 121):
        previous = env.monster_pos
        _, _, done, truncated, _ = env.step(3)
        assert env.monster_pos in env.floor_cells
        distance = sum(abs(a - b) for a, b in zip(previous, env.monster_pos))
        assert distance == (1 if step % 2 == 0 else 0)
        if done or truncated:
            break
    env.close()


def test_agent_into_monster_is_fatal_before_patrol(fixed):
    base = fixed.unwrapped
    place_monster(base, (3, 2))
    base.monster_interval = 1
    _, _, terminated, truncated, info = fixed.step(2)
    assert terminated and not truncated
    assert base.monster_pos == (3, 2)
    assert info["reward_terms"]["death"] == -1
    with pytest.raises(RuntimeError, match="reset"):
        fixed.step(3)


def test_monster_into_agent_is_fatal(fixed):
    base = fixed.unwrapped
    place_monster(base, (3, 2))
    base.monster_interval = 1
    # Restrict the monster's legal neighborhood to the agent cell.
    for p in ((4, 2), (3, 1), (3, 3)):
        base.floor_cells.remove(p)
        base.grid.set(*p, Wall())
    _, _, terminated, truncated, _ = fixed.step(3)
    assert terminated and not truncated
    assert base.monster_pos == (2, 2)


def test_time_limit_is_truncation():
    env = make_env(max_steps=1, monster_interval=100)
    env.reset(seed=2)
    _, _, terminated, truncated, _ = env.step(3)
    assert truncated and not terminated
    env.close()


def test_visibility_blocked_by_solid_wall(fixed):
    base = fixed.unwrapped
    for y in range(1, 8):
        base.grid.set(4, y, Wall())
    original = base.gen_obs()["image"].copy()
    base.grid.set(6, 2, Floor("green"))
    np.testing.assert_array_equal(original, base.gen_obs()["image"])


def test_hidden_changes_do_not_affect_observation_or_reward():
    a = ExplorationWrapper(FixtureEnv(size=9, monster_interval=10000))
    b = ExplorationWrapper(FixtureEnv(size=9, monster_interval=10000))
    a.reset(seed=4)
    b.reset(seed=4)
    # Behind the agent, outside its forward view.
    b.unwrapped.grid.set(1, 1, Floor("green"))
    place_monster(b.unwrapped, (1, 7))
    ra, rb = a.step(3), b.step(3)
    for k in ra[0]:
        np.testing.assert_array_equal(ra[0][k], rb[0][k])
    assert ra[1:] == rb[1:]
    assert set(ra[4]) == {"reward_terms"}
    a.close()
    b.close()


def test_reset_clears_episode_memory(fixed):
    fixed.step(2)
    fixed.step(2)
    fixed.reset(seed=7)
    assert fixed.memory.position == (27, 27)
    assert np.count_nonzero(fixed.memory.visits) == 1
    assert fixed.total_reward == 0


def test_rewards_not_farmed_by_waiting_turning_or_revisits(fixed):
    for _ in range(8):
        fixed.step(1)
    before = set(fixed.memory.discovered_poi)
    for _ in range(4):
        _, reward, _, _, info = fixed.step(1)
        assert reward == pytest.approx(-0.001)
        assert info["reward_terms"]["seen"] == 0
        assert info["reward_terms"]["poi"] == 0
    _, reward, _, _, _ = fixed.step(3)
    assert reward == pytest.approx(-0.001)
    assert fixed.memory.discovered_poi == before


def test_reward_formula_and_new_visit_once(fixed):
    _, reward, _, _, info = fixed.step(2)
    terms = info["reward_terms"]
    assert terms["visited"] == 0.01
    assert reward == pytest.approx(sum(terms.values()))
    fixed.step(0)
    fixed.step(0)
    _, _, _, _, info = fixed.step(2)
    assert info["reward_terms"]["visited"] == 0


def test_poi_discovery_awarded_once_after_initialization(fixed):
    base = fixed.unwrapped
    # Add a POI to a previously observed floor: next observation is a first POI sighting.
    base.grid.set(3, 3, Floor("green"))
    _, _, _, _, first = fixed.step(3)
    _, _, _, _, second = fixed.step(3)
    assert first["reward_terms"]["poi"] == 1
    assert second["reward_terms"]["poi"] == 0


def test_no_memory_ablation_keeps_internal_reward_tracking():
    env = make_env(variant="no_memory")
    obs, _ = env.reset(seed=4)
    assert not np.any(obs["memory"])
    assert env.memory.seen_floor
    env.close()


@pytest.mark.parametrize("kwargs", [{"size": 28}, {"size": 8}, {"poi_count": 0}, {"monster_interval": 0}, {"max_steps": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ExplorationEnv(**kwargs)


def test_invalid_action(fixed):
    with pytest.raises(ValueError):
        fixed.step(4)
