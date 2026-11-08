from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image

from exploration.agents import create_agent
from exploration.evaluation import episode_metrics
from exploration.memory import make_env
from exploration.rendering import episode_frame


def replay(agent_name, checkpoint, seed, preset, output, *, frame_stride=4, fps=12):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    agent = create_agent(agent_name, checkpoint)
    metadata = getattr(agent, "metadata", {})
    env = make_env(preset, variant=metadata.get("variant", "main"),
                   reward=metadata.get("config", {}).get("reward"))
    obs, _ = env.reset(seed=seed)
    agent.reset(seed)
    actions = []
    terminated = truncated = False
    title = f"{agent.name.upper()} / {preset.upper()}"
    subtitle = f"Map seed {seed} | Partial 7 x 7 observations | Green = POI | Red = moving threat"
    writer = imageio.get_writer(output / "episode.mp4", fps=fps, codec="libx264", quality=7,
                                macro_block_size=1, ffmpeg_log_level="error")
    try:
        writer.append_data(episode_frame(env, title, subtitle))
        while not (terminated or truncated):
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            actions.append(action)
            if len(actions) % frame_stride == 0 or terminated or truncated:
                writer.append_data(episode_frame(env, title, subtitle))
        metrics = episode_metrics(env, terminated, truncated)
        final = episode_frame(env, title, subtitle)
        Image.fromarray(final).save(output / "final.png")
        trace = {"agent": agent.name, "seed": seed, "preset": preset,
                 "checkpoint": str(checkpoint) if checkpoint else None,
                 "checkpoint_sha256": metadata.get("sha256"), "variant": env.variant,
                 "memory_frame": getattr(agent, "memory_frame", "start_aligned"),
                 "frame_stride": frame_stride, "fps": fps,
                 "actions": actions, "metrics": metrics}
        (output / "replay.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
        return trace
    finally:
        writer.close()
        env.close()
