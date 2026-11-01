import numpy as np

from exploration.memory import make_env
from scripts.investigate_memory import Audit, transform


def test_current_view_memory_matches_native_visibility_without_mutation():
    env = make_env("easy")
    obs, _ = env.reset(seed=10000)
    original = obs["memory"].copy()
    current = transform(obs, "no_history")["memory"]
    x, y = env.memory.position
    origin = tuple(env.unwrapped.agent_pos)
    for my, mx in np.argwhere(current[0] > 0):
        wx, wy = mx - x + origin[0], my - y + origin[1]
        vx, vy = env.unwrapped.get_view_coords(wx, wy)
        assert 0 <= vx < 7 and 0 <= vy < 7
        assert obs["local"][0, vy, vx] == 0
    assert np.count_nonzero(current[0]) == np.count_nonzero(obs["local"][0] == 0)
    assert not np.any(current[4])
    np.testing.assert_array_equal(obs["memory"], original)
    env.close()


def test_map_translation_preserves_all_channel_mass_and_relative_geometry():
    env = make_env("easy")
    obs, _ = env.reset(seed=10001)
    shifted = transform(obs, "translate")
    np.testing.assert_array_equal(obs["memory"].sum(axis=(1, 2)), shifted["memory"].sum(axis=(1, 2)))
    before = np.argwhere(obs["memory"][0] > 0)
    after = np.argwhere(shifted["memory"][0] > 0)
    differences = after - before
    assert np.all(differences == differences[0])
    np.testing.assert_array_equal(shifted["scalars"], obs["scalars"])
    env.close()


def test_four_rotations_restore_complete_observation():
    env = make_env("medium")
    obs, _ = env.reset(seed=10002)
    rotated = obs
    for _ in range(4):
        rotated = transform(rotated, "rotate90")
    for key in obs:
        np.testing.assert_array_equal(rotated[key], obs[key])
    env.close()


def test_mapping_audit_detects_missing_current_and_forgotten_cells():
    env = make_env("easy")
    env.reset(seed=10000)
    origin = tuple(env.unwrapped.agent_pos)
    audit = Audit()
    audit.check(env, origin)
    assert sum(audit.errors.values()) == 0
    # Deliberately corrupt an observed cell; the audit must detect the omission.
    y, x = np.argwhere(env.memory.map[0] > 0)[0]
    env.memory.map[0, y, x] = 0
    audit.check(env, origin)
    assert audit.errors["missing_current_visible"] > 0
    assert audit.errors["forgotten_observations"] > 0
    env.close()


def test_mapping_audit_detects_loss_outside_current_view():
    env = make_env("easy")
    initial, _ = env.reset(seed=10000)
    audit = Audit()
    origin = tuple(env.unwrapped.agent_pos)
    audit.check(env, origin)
    env.step(0)
    observation, _, _, _, _ = env.step(0)
    current = transform(observation, "no_history")["memory"]
    historical = (initial["memory"][0] > 0) & (current[0] == 0)
    assert historical.any()
    y, x = np.argwhere(historical)[0]
    env.memory.map[0, y, x] = 0
    audit.check(env, origin)
    assert audit.errors["missing_current_visible"] == 0
    assert audit.errors["forgotten_observations"] == 1
    env.close()
