"""Memory alignment geometry, compatibility, provenance, and regression tests.

Checkpoint fixtures construct UNTRAINED policies. Nothing in this file calls
learn(), launches jobs, or modifies shipped checkpoints/results.
"""
import csv
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from exploration.__main__ import parser, training_config
from exploration.agents import FrontierAgent, PPOAgent
from exploration.config import (
    checkpoint_memory_frame,
    load_config,
    policy_label,
    validate_config,
)
from exploration.memory import make_env
from exploration.policy import (
    ExplorationFeatures,
    align_memory,
    load_checkpoint,
    make_model,
    save_checkpoint,
)
from exploration.training import validate_resume


@pytest.fixture(params=("cpu", "cuda"))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def observation_space():
    return spaces.Dict({
        "local": spaces.Box(0, 1, (5, 7, 7), np.float32),
        "memory": spaces.Box(0, 1, (7, 55, 55), np.float32),
        "scalars": spaces.Box(0, 1, (8,), np.float32),
    })


@pytest.mark.parametrize("heading", range(4))
@pytest.mark.parametrize("position", ((27, 27), (10, 20), (43, 12)))
def test_cardinal_cells_and_marker_have_known_positions(device, heading, position):
    x, y = position
    memory = torch.zeros((1, 7, 55, 55), device=device)
    memory[0, 6, y, x] = 1
    # Fixed world EAST and NORTH cells; expected positions are hand specified.
    memory[0, 0, y, x + 1] = 0.25
    memory[0, 1, y - 1, x] = 0.5
    east_xy = ((27, 26), (26, 27), (27, 28), (28, 27))[heading]
    north_xy = ((26, 27), (27, 28), (28, 27), (27, 26))[heading]
    original = memory.clone()
    aligned = align_memory(memory, torch.tensor([heading], device=device))
    expected = torch.zeros_like(memory)
    expected[0, 6, 27, 27] = 1
    expected[0, 0, east_xy[1], east_xy[0]] = 0.25
    expected[0, 1, north_xy[1], north_xy[0]] = 0.5
    torch.testing.assert_close(aligned, expected, rtol=0, atol=0)
    torch.testing.assert_close(memory, original, rtol=0, atol=0)
    assert aligned.data_ptr() != memory.data_ptr()


@pytest.mark.parametrize("heading", range(4))
@pytest.mark.parametrize("position", ((0, 0), (54, 0), (0, 54), (54, 54)))
def test_boundary_padding_copies_all_channels_without_wrapping(device, heading, position):
    memory = torch.zeros((1, 7, 55, 55), device=device)
    for channel in range(6):
        memory[0, channel] = (channel + 1) / 10
    x, y = position
    memory[0, 6, y, x] = 1
    aligned = align_memory(memory, torch.tensor([heading], device=device))
    valid = aligned[0, 0] > 0
    assert valid.sum().item() == 28 * 28
    for channel in range(6):
        torch.testing.assert_close(aligned[0, channel], valid * ((channel + 1) / 10))
    assert aligned[0, 6, 27, 27].item() == 1
    assert aligned[0, 6].sum().item() == 1


def patterned_map(device):
    memory = torch.zeros((1, 7, 55, 55), device=device)
    memory[0, 6, 18, 12] = 1
    for channel in range(6):
        memory[0, channel, 16:21, 10:15] = (channel + 1) / 10
    return memory


@pytest.mark.parametrize("heading", range(4))
def test_joint_translation_and_rotation_are_invariant(device, heading):
    memory = patterned_map(device)
    h = torch.tensor([heading], device=device)
    expected = align_memory(memory, h)
    translated = torch.roll(memory, (4, -3), dims=(-2, -1))
    rotated = torch.rot90(memory, 1, dims=(-2, -1))
    torch.testing.assert_close(align_memory(translated, h), expected, rtol=0, atol=0)
    torch.testing.assert_close(align_memory(rotated, (h - 1) % 4), expected, rtol=0, atol=0)


def test_batch_zero_maps_and_gradients(device):
    memories = [patterned_map(device), torch.zeros((1, 7, 55, 55), device=device),
                torch.roll(patterned_map(device), (3, 5), dims=(-2, -1))]
    headings = torch.tensor([0, 2, 3], device=device)
    memory = torch.cat(memories).requires_grad_()
    result = align_memory(memory, headings)
    individually = torch.cat([align_memory(m, h[None]) for m, h in zip(memories, headings)])
    torch.testing.assert_close(result, individually, rtol=0, atol=0)
    assert result[1].count_nonzero().item() == 0
    result.sum().backward()
    assert torch.isfinite(memory.grad).all()


@pytest.mark.parametrize("size", (15, 21, 27))
@pytest.mark.parametrize("heading", range(4))
def test_supported_terrain_survives_centering(size, heading):
    # Known terrain occupies a size x size world square; the agent is at its
    # opposite corners. Maximum distance is 26, within the 27-cell radius.
    for x, y in ((14, 14), (14 + size - 1, 14 + size - 1)):
        memory = torch.zeros((1, 7, 55, 55))
        memory[0, 0, 14:14 + size, 14:14 + size] = 1
        memory[0, 6, y, x] = 1
        aligned = align_memory(memory, torch.tensor([heading]))
        torch.testing.assert_close(aligned.sum(dim=(-2, -1)), memory.sum(dim=(-2, -1)))


def test_extractor_frames_preserve_inputs_parameters_and_legacy_math():
    legacy = ExplorationFeatures(observation_space())
    aligned = ExplorationFeatures(observation_space(), memory_frame="agent_aligned")
    aligned.load_state_dict(legacy.state_dict(), strict=True)
    assert sum(p.numel() for p in aligned.parameters()) == sum(p.numel() for p in legacy.parameters())
    assert set(aligned.state_dict()) == set(legacy.state_dict())
    obs = {"local": torch.zeros(1, 5, 7, 7), "memory": patterned_map("cpu"),
           "scalars": torch.tensor([[1., 0, 0, 0, 0, 0, 0, 1]])}
    original = {key: value.clone() for key, value in obs.items()}
    expected_legacy = torch.cat((legacy.local(obs["local"]), legacy.memory(obs["memory"]), obs["scalars"]), dim=1)
    expected_aligned = torch.cat((aligned.local(obs["local"]),
                                  aligned.memory(align_memory(obs["memory"], torch.tensor([0]))),
                                  obs["scalars"]), dim=1)
    torch.testing.assert_close(legacy(obs), expected_legacy, rtol=0, atol=0)
    torch.testing.assert_close(aligned(obs), expected_aligned, rtol=0, atol=0)
    assert aligned(obs).shape == (1, 200)
    for key, value in obs.items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


@pytest.fixture
def untrained_checkpoint(tmp_path):
    def create(frame="start_aligned", legacy=False):
        config = load_config()
        config.update(device="cpu", output=str(tmp_path), memory_frame=frame)
        env = make_env()
        try:
            model = make_model(env, config)
            if legacy:
                # Emulate the constructor options in pre-redesign SB3 archives.
                model.policy_kwargs.pop("features_extractor_kwargs", None)
                model.policy.features_extractor_kwargs.clear()
            path = save_checkpoint(model, tmp_path / f"{frame}.zip", config)
            if legacy:
                metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                metadata.pop("memory_frame")
                metadata["config"].pop("memory_frame")
                path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            return model, path
        finally:
            env.close()
    return create


@pytest.mark.parametrize("frame", ("start_aligned", "agent_aligned"))
def test_untrained_checkpoint_roundtrip_and_labels(untrained_checkpoint, frame):
    model, path = untrained_checkpoint(frame)
    loaded, metadata = load_checkpoint(path)
    assert metadata["training_steps"] == 0
    assert metadata["memory_frame"] == frame
    assert loaded.policy.features_extractor.memory_frame == frame
    for key, value in model.policy.state_dict().items():
        torch.testing.assert_close(loaded.policy.state_dict()[key], value, rtol=0, atol=0)
    agent = PPOAgent(path)
    assert agent.name == policy_label("main", frame)
    assert agent.memory_frame == frame
    # A neighboring final sidecar must not inflate counts across memory frames.
    other = deepcopy(metadata)
    other.update(memory_frame="agent_aligned" if frame == "start_aligned" else "start_aligned", training_steps=99)
    other["config"]["memory_frame"] = other["memory_frame"]
    path.with_name("final.json").write_text(json.dumps(other), encoding="utf-8")
    assert PPOAgent(path).completed_training_steps == 0


def test_complete_policy_parameter_structure_is_unchanged(untrained_checkpoint):
    original, _ = untrained_checkpoint("start_aligned")
    aligned, _ = untrained_checkpoint("agent_aligned")
    assert sum(p.numel() for p in original.policy.parameters()) == sum(p.numel() for p in aligned.policy.parameters())
    assert {k: tuple(v.shape) for k, v in original.policy.state_dict().items()} == {
        k: tuple(v.shape) for k, v in aligned.policy.state_dict().items()}


def test_legacy_archive_and_metadata_default_to_start_frame(untrained_checkpoint):
    _, path = untrained_checkpoint(legacy=True)
    model, metadata = load_checkpoint(path)
    assert metadata["memory_frame"] == "start_aligned"
    assert model.policy.features_extractor.memory_frame == "start_aligned"
    assert PPOAgent(path).name == "main"


@pytest.mark.parametrize("mismatch", ("metadata_config", "extractor"))
def test_checkpoint_frame_mismatches_rejected(untrained_checkpoint, mismatch):
    _, path = untrained_checkpoint("agent_aligned")
    sidecar = path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["memory_frame"] = "start_aligned"
    if mismatch == "extractor":
        metadata["config"]["memory_frame"] = "start_aligned"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="memory frame"):
        load_checkpoint(path)


def test_live_extractor_mismatch_is_rejected(untrained_checkpoint, tmp_path):
    model, _ = untrained_checkpoint("start_aligned")
    model.policy.features_extractor.memory_frame = "agent_aligned"
    with pytest.raises(ValueError, match="memory frame"):
        save_checkpoint(model, tmp_path / "invalid.zip", load_config())


def test_save_rejects_mismatched_config_before_writing(untrained_checkpoint, tmp_path):
    model, _ = untrained_checkpoint("agent_aligned")
    config = load_config()
    with pytest.raises(ValueError, match="memory frame"):
        save_checkpoint(model, tmp_path / "invalid.zip", config)
    assert not (tmp_path / "invalid.zip").exists()


@pytest.mark.parametrize("changes", ({"seed": 22}, {"variant": "no_memory"}, {"memory_frame": "agent_aligned"}))
def test_resume_checks_identity_and_frame(changes):
    config = load_config()
    metadata = {"seed": 11, "variant": "main", "config": {}}
    validate_resume(metadata, config)
    config.update(changes)
    with pytest.raises(ValueError, match="Resume"):
        validate_resume(metadata, config)


@pytest.mark.parametrize("options,output", (
    ([], "runs/main-11"),
    (["--memory-frame", "agent_aligned"], "runs/main_agent_aligned-11"),
    (["--memory-frame", "agent_aligned", "--seed", "22"], "runs/main_agent_aligned-22"),
    (["--memory-frame", "agent_aligned", "--output", "custom"], "custom"),
    (["--variant", "no_memory", "--seed", "33"], "runs/no_memory-33"),
))
def test_cli_config_and_distinct_output_names(options, output):
    config = training_config(parser().parse_args(["train", *options]))
    assert config["output"] == output


def test_opt_in_yaml_matches_main_experiment_except_frame_and_output():
    root = Path(__file__).resolve().parents[1]
    original = load_config(root / "configs/main.yaml")
    aligned = load_config(root / "configs/memory-aligned.yaml")
    assert aligned["memory_frame"] == "agent_aligned"
    assert aligned["output"] == "runs/main_agent_aligned-11"
    aligned.update(memory_frame=original["memory_frame"], output=original["output"])
    assert aligned == original


def test_config_validation_and_legacy_metadata():
    assert checkpoint_memory_frame({}) == "start_aligned"
    config = load_config()
    config["memory_frame"] = "unsupported"
    with pytest.raises(ValueError, match="memory frame"):
        validate_config(config)


def test_frame_only_yaml_gets_an_isolated_default_folder(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("memory_frame: agent_aligned\nseed: 22\n", encoding="utf-8")
    assert load_config(path)["output"] == "runs/main_agent_aligned-22"


def test_evaluation_records_frame_in_rows_curves_and_provenance(untrained_checkpoint, tmp_path):
    from exploration.evaluation import evaluate

    _, path = untrained_checkpoint("agent_aligned")
    output = tmp_path / "evaluation"
    report = evaluate([PPOAgent(path)], ["easy"], [10000], output, env_kwargs={"max_steps": 1})
    assert report["agents"][0]["memory_frame"] == "agent_aligned"
    assert "main_agent_aligned/easy" in report["summary"]
    for filename in ("episodes.csv", "curves.csv"):
        with (output / filename).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows and all(row["memory_frame"] == "agent_aligned" for row in rows)


def test_replay_trace_records_frame_without_encoding_video(untrained_checkpoint, tmp_path, monkeypatch):
    import exploration.replay as replay_module

    _, path = untrained_checkpoint("agent_aligned")
    monkeypatch.setattr(replay_module, "make_env", lambda preset, **kwargs: make_env(preset, max_steps=1, **kwargs))
    writer = SimpleNamespace(append_data=lambda frame: None, close=lambda: None)
    monkeypatch.setattr(replay_module.imageio, "get_writer", lambda *args, **kwargs: writer)
    monkeypatch.setattr(replay_module, "episode_frame", lambda *args, **kwargs: np.zeros((4, 4, 3), dtype=np.uint8))
    trace = replay_module.replay("ppo", path, 10000, "easy", tmp_path / "replay")
    assert trace["memory_frame"] == "agent_aligned"
    assert trace["agent"] == "main_agent_aligned"


def test_wrapper_reward_history_and_frontier_are_unaffected():
    # Integration regression for later execution. Calling the transformer must
    # not alter observations, bookkeeping, rewards, or baseline decisions.
    left, right = make_env("easy"), make_env("easy")
    try:
        a, _ = left.reset(seed=10000)
        b, _ = right.reset(seed=10000)
        frontier = FrontierAgent()
        for action in (0, 2, 1, 2, 3, 0, 0, 2):
            before = left.memory.map.copy()
            aligned = align_memory(torch.from_numpy(a["memory"])[None],
                                   torch.tensor([int(a["scalars"][:4].argmax())]))
            assert aligned.shape == (1, 7, 55, 55)
            np.testing.assert_array_equal(left.memory.map, before)
            assert frontier.act(a) == frontier.act(b)
            a, reward_a, terminated_a, truncated_a, info_a = left.step(action)
            b, reward_b, terminated_b, truncated_b, info_b = right.step(action)
            assert (reward_a, terminated_a, truncated_a, info_a) == (reward_b, terminated_b, truncated_b, info_b)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            if terminated_a or truncated_a:
                break
        left.reset(seed=10001)
        assert left.memory.position == (27, 27)
        assert np.count_nonzero(left.memory.visits) == 1
    finally:
        left.close()
        right.close()
