"""Produce a four-minute aligned-policy demo without altering the frozen study.

The checkpoint is selected by validation score. Qualitative maps are declared
below; recorded benchmark means are never recomputed from these examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.memory import make_env
from exploration.rendering import font
from exploration.replay import replay

FPS = 12
SIZE = (1280, 720)
STUDY = ROOT / "results/aligned-reward-followup"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slide(title, lines, footer="", picture=None):
    canvas = Image.new("RGB", SIZE, "#0b1120")
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 24), "MINIGRID / ALIGNED EXPLORATION", font=font(19, True), fill="#74d4c1")
    draw.text((42, 70), title, font=font(34, True), fill="#f2f5fb")
    y = 137
    for line in lines:
        draw.text((42, y), line, font=font(24), fill="#dbe5f3")
        y += 39
    if picture is not None:
        image = Image.open(picture).convert("RGB")
        image.thumbnail((1196, 645 - y), Image.Resampling.LANCZOS)
        canvas.paste(image, ((1280 - image.width) // 2, y + 8))
    draw.text((42, 678), footer, font=font(17), fill="#93a4bd")
    return np.asarray(canvas)


def prepare(output):
    import torch

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((ROOT / "checkpoints/aligned/index.json").read_text())
    selected = max(index, key=lambda x: tuple(x["validation_score"]))
    checkpoint = ROOT / "checkpoints/aligned" / selected["checkpoint"]
    assert digest(checkpoint) == selected["sha256"]
    assert selected["seed"] == 33 and selected["memory_frame"] == "agent_aligned"
    cases = [("aligned-easy", "ppo", checkpoint, 10000, "easy"),
             ("aligned-medium", "ppo", checkpoint, 10000, "medium"),
             ("aligned-hard", "ppo", checkpoint, 10000, "hard"),
             ("aligned-hard-typical", "ppo", checkpoint, 10039, "hard"),
             ("frontier-medium", "frontier", None, 10000, "medium"),
             ("aligned-contact", "ppo", checkpoint, 30000, "hard"),
             ("loop-example-seed11", "ppo", ROOT / "checkpoints/aligned/main_agent_aligned-11.zip", 30029, "hard")]
    traces = {}
    for name, agent, model, seed, preset in cases:
        folder = output / "replays" / name
        trace_path = folder / "replay.json"
        expected_hash = digest(model) if model else None
        historical = ROOT / "submission/aligned/replays" / name
        if not trace_path.exists() and (historical / "replay.json").exists():
            shutil.copytree(historical, folder, dirs_exist_ok=True)
        if trace_path.exists():
            trace = json.loads(trace_path.read_text())
            assert trace["checkpoint_sha256"] == expected_hash and trace["seed"] == seed and trace["preset"] == preset
            assert (folder / "episode.mp4").is_file() and (folder / "final.png").is_file()
        else:
            trace = replay(agent, str(model) if model else None, seed, preset, folder,
                           frame_stride=1 if name == "aligned-hard-typical" else 4)
        traces[name] = trace

    selection_dir = ROOT / "submission/final/hard-example-selection"
    selection = json.loads((selection_dir / "selection.json").read_text())
    audit_trace = json.loads((selection_dir / "selected-trace.json").read_text())
    assert selection["checkpoint_sha256"] == selected["sha256"]
    assert selection["example_seed"] == 10039 and selection["example_is_nearest_median_candidate"]
    assert traces["aligned-hard-typical"]["actions"] == audit_trace["actions"]
    assert traces["aligned-hard-typical"]["frame_stride"] == 1
    assert traces["aligned-hard-typical"]["metrics"] == audit_trace["metrics"]

    # Procedural-generation panel uses fixed seeds; representative Hard replay is separately disclosed.
    canvas = Image.new("RGB", SIZE, "#0b1120")
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 24), "MINIGRID / PROCEDURAL ENVIRONMENT", font=font(19, True), fill="#74d4c1")
    draw.text((42, 70), "Connected rooms, corridors, and a moving patrol", font=font(32, True), fill="#f2f5fb")
    for i, preset in enumerate(("easy", "medium", "hard")):
        env = make_env(preset)
        try:
            env.reset(seed=10000)
            image = Image.fromarray(env.unwrapped.get_frame(tile_size=20))
            image.thumbnail((365, 430), Image.Resampling.NEAREST)
            x = 35 + i * 420
            canvas.paste(image, (x + (365 - image.width) // 2, 190 + (430 - image.height) // 2))
            base = env.unwrapped
            label = f"{preset.title()}: {base.width} x {base.height}, {base.poi_count} POIs"
            draw.text((x, 145), label, font=font(23, True), fill="#dbe5f3")
            draw.text((x, 625), f"Budget {base.max_steps} | seed 10000", font=font(20), fill="#bdcbe0")
        finally:
            env.close()
    draw.text((42, 678), "Evaluator views only. All presets reuse the same wall-occluded 7 x 7 agent interface.", font=font(17), fill="#93a4bd")
    canvas.save(output / "generation.png")

    summary = json.loads((STUDY / "summary.json").read_text())
    canvas = Image.fromarray(slide("Exploration, discovery, and survival", [],
                                  "5,700 episodes | 100 reserved maps per preset | Three PPO training seeds | Intervals in results archive"))
    draw = ImageDraw.Draw(canvas)
    columns = [42, 550, 730, 920, 1100]
    for x, label in zip(columns, ("Agent", "Coverage", "Visited", "POIs", "Survival")):
        draw.text((x, 145), label, font=font(24, True), fill="#93a4bd")
    order = [("random", "Random"), ("frontier", "Frontier (BFS)"), ("no_novelty", "PPO, no bonuses (original map)"),
             ("no_memory", "PPO, no map"), ("main", "PPO, original map"),
             ("no_novelty_agent_aligned", "PPO, no bonuses (aligned)"),
             ("main_agent_aligned", "PPO, aligned map")]
    for i, (key, label) in enumerate(order):
        y = 202 + i * 45
        if key == "main_agent_aligned":
            draw.rounded_rectangle((30, y - 6, 1246, y + 42), radius=5, fill="#173c41")
        values = summary["overall"][key]
        cells = [label] + [f"{100 * values[m]:.1f}%" for m in ("revealed_coverage", "visited_coverage", "poi_fraction", "survived")]
        for x, value in zip(columns, cells):
            draw.text((x, y), value, font=font(23, key == "main_agent_aligned"), fill="#f2f5fb")
    draw.text((42, 550), "Frontier explores faster; aligned PPO survives more often than frontier.", font=font(23), fill="#74d4c1")
    draw.text((42, 592), "Hard: aligned coverage 81.2% vs mapless 63.5%; survival 62.7% vs 70.7%.", font=font(23), fill="#efbe76")
    canvas.save(output / "comparison.png")
    metadata = {"checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": selected["sha256"],
                "training_seed": selected["seed"], "selected_steps": selected["selected_steps"],
                "memory_frame": selected["memory_frame"], "validation_score": selected["validation_score"],
                "selection": "Highest lexicographic validation AUC/discovery/survival score among the three aligned runs",
                "qualitative_note": "Easy, Medium, and frontier use map 10000. Main Hard replay uses map 10039, one of two equally near the 50-map Hard validation coverage median, selected for a survived close pass. The same-model map-10000 turning loop is retained as a failure example. Contact map 30000 is a selected illustration. Stored seed-11/map-30029 replay is not shown in this revision. Clips are qualitative, not aggregate evidence.",
                "comparison_seed_disclosure": "Primary: predeclared 40000-40099, no checkpoint selection on confirmation. Older 20000/30000 ranges exposed during diagnosis; archived evidence and disclosure retained.",
                "study_summary_sha256": digest(STUDY / "summary.json"),
                "case_metrics": {key: trace["metrics"] for key, trace in traces.items()}}
    metadata["presentation_revision"] = {
        "date": "2026-11-13",
        "changes": "Near-median Hard exploration and a survived close pass, with same-model turning-loop failure retained",
        "source_replays_unchanged": True, "release_tag": "v1.0.2"}
    metadata["release_tag"] = "v1.0.2"
    metadata["report_included"] = False
    metadata["shown_replays"] = ["aligned-easy", "aligned-medium", "aligned-hard-typical", "frontier-medium", "aligned-contact", "aligned-hard"]
    metadata["hard_loop_diagnostics"] = "submission/final/hard-replay-diagnostics.json"
    metadata["hard_example_selection"] = {
        "record": str((selection_dir / "selection.json").relative_to(ROOT)),
        "record_sha256": digest(selection_dir / "selection.json"),
        "episodes_sha256": digest(selection_dir / "episodes.csv"),
        "median_coverage": selection["median_coverage"], "example_seed": 10039,
        "nearest_median_candidates": selection["nearest_median_candidates"],
        "rule": "Near-median validation coverage; of two equally close episodes, use the survived close-pass example.",
        "not_checkpoint_selection": True,
        "proximity_definition": "Post-step Manhattan distance of one cell; no claim of intentional avoidance or eight distinct encounters"}
    (output / "demo-provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def render(output, ui_video):
    metadata = json.loads((output / "demo-provenance.json").read_text())
    pending_video = output / "demo.rendering.mp4"
    writer = imageio.get_writer(pending_video, fps=FPS, codec="libx264", quality=7,
                                macro_block_size=1, ffmpeg_log_level="error")
    storyboard = []
    elapsed = 0
    preview_names = set()

    def still(name, frame, seconds):
        nonlocal elapsed
        storyboard.append({"start": elapsed, "seconds": seconds, "title": name})
        preview_name = f"{elapsed:03d}.png"
        preview_names.add(preview_name)
        Image.fromarray(frame).save(output / "previews" / preview_name)
        for _ in range(seconds * FPS):
            writer.append_data(frame)
        elapsed += seconds

    def clip(name, path, seconds, caption=None, note=None, phases=None, highlight=None):
        nonlocal elapsed
        reader = imageio.get_reader(path)
        try:
            length = reader.count_frames()
            ui_crop = None
            if Path(path).resolve() == Path(ui_video).resolve():
                first = reader.get_data(0)
                ys, xs = np.nonzero(np.max(first, axis=2) > 10)
                ui_crop = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            trace_path = Path(path).with_name("replay.json")
            trace = json.loads(trace_path.read_text()) if trace_path.exists() else None
            moving_frames = (seconds - 2) * FPS
            timing = []
            if phases:
                assert trace and sum(p[2] for p in phases) == seconds - 2
                stride = trace["frame_stride"]
                # Recorded source frames are step zero, every stride, and the final step.
                source_steps = np.minimum(np.arange(length) * stride, len(trace["actions"]))
                parts, messages = [], []
                phase_start = 0
                for first_step, last_step, duration, message in phases:
                    source_indices = np.interp([first_step, last_step], source_steps, np.arange(length))
                    part = np.linspace(*source_indices, duration * FPS).astype(int)
                    parts.append(part)
                    messages.extend([message] * len(part))
                    timing.append({"source_steps": [first_step, last_step], "screen_start": elapsed + phase_start,
                                   "screen_seconds": duration, "caption": message})
                    phase_start += duration
                indices = np.concatenate(parts + [np.full(2 * FPS, length - 1, dtype=int)])
                messages.extend(["Final outcome / held for 2 seconds"] * (2 * FPS))
            else:
                indices = np.concatenate((np.linspace(0, length - 1, moving_frames).astype(int),
                                          np.full(2 * FPS, length - 1, dtype=int)))
                messages = [note] * moving_frames + ["Final outcome / held for 2 seconds"] * (2 * FPS)
            assert len(indices) == seconds * FPS and np.all(np.diff(indices) >= 0)
            assert indices[0] == 0 and indices[-1] == length - 1
            for k, index in enumerate(indices):
                frame = reader.get_data(int(index))
                if ui_crop:
                    x0, y0, x1, y1 = ui_crop
                    frame = frame[y0:y1, x0:x1]
                if trace:
                    canvas = Image.fromarray(frame).convert("RGB")
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((0, 0, 1280, 100), fill="#0b1120")
                    draw.text((38, 25), name, font=font(29, True), fill="#f2f5fb")
                    seed = Path(trace["checkpoint"]).stem.rsplit("-", 1)[-1] if trace["checkpoint"] else None
                    policy = "Frontier (BFS)" if trace["agent"] == "frontier" else f"Aligned PPO / training seed {seed}"
                    if phases:
                        playback = "Variable-speed replay / close pass slowed; loops fast-forwarded" if highlight else "Variable-speed replay / turning loop fast-forwarded"
                    else:
                        playback = "Episode replay / compressed playback"
                    draw.text((38, 70), f"Map {trace['seed']} | {policy} | {playback}",
                              font=font(17), fill="#93a4bd")
                    if highlight:
                        source_step = min(int(index) * trace["frame_stride"], len(trace["actions"]))
                        if highlight["first_step"] <= source_step <= highlight["last_step"]:
                            state = highlight["states"].get(source_step)
                            if state:
                                # Rings annotate evaluator-only positions on the 27x27 ground-truth panel.
                                for point in (state["agent"], state["monster"]):
                                    cx = 60 + (point[0] + 0.5) * 495 / 27
                                    cy = 140 + (point[1] + 0.5) * 495 / 27
                                    draw.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), outline="#ffd469", width=3)
                    if messages[k] and not caption:
                        draw.rectangle((0, 636, 1280, 660), fill="#0b1120")
                        color = "#efbe76" if "loop" in messages[k].lower() else "#74d4c1"
                        draw.text((38, 637), messages[k], font=font(18), fill=color)
                    frame = np.asarray(canvas)
                if caption:
                    canvas = Image.new("RGB", SIZE, "#0b1120")
                    image = Image.fromarray(frame).convert("RGB")
                    image.thumbnail((1232, 624), Image.Resampling.LANCZOS)
                    canvas.paste(image, ((1280 - image.width) // 2, (624 - image.height) // 2))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((28, 636), name, font=font(24, True), fill="#74d4c1")
                    draw.text((28, 679), caption, font=font(18), fill="#dbe5f3")
                    frame = np.asarray(canvas)
                if k in (0, seconds * FPS // 2, seconds * FPS - 1):
                    preview_name = f"{elapsed:03d}-{k:03d}.png"
                    preview_names.add(preview_name)
                    Image.fromarray(frame).save(output / "previews" / preview_name)
                writer.append_data(frame)
        finally:
            reader.close()
        storyboard.append({"start": elapsed, "seconds": seconds, "title": name,
                           "source": str(Path(path).relative_to(ROOT)), "ui_capture_padding_crop": ui_crop,
                           "source_sha256": digest(path), "final_hold_seconds": 2,
                           "presentation": "Variable-speed episode replay; turning loops labeled and fast-forwarded" if phases else "Whole source clip resampled; final outcome held for two seconds",
                           "timing_segments": timing,
                           "evaluator_highlight": {"source_steps": [highlight["first_step"], highlight["last_step"]],
                                                   "meaning": "Gold rings mark ground-truth agent and monster positions; not policy inputs"} if highlight else None,
                           "source_frame_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()})
        elapsed += seconds

    (output / "previews").mkdir(exist_ok=True)
    audit = json.loads((ROOT / "submission/final/hard-example-selection/selected-trace.json").read_text())
    highlight = {"first_step": 250, "last_step": 268,
                 "states": {state["step"]: state for state in audit["states_after_step"]}}
    assert highlight["states"][258]["manhattan_distance"] == 1
    hard_phases = [
        (0, 244, 8, "Chosen near the median of 50 Hard validation episodes / fixed checkpoint."),
        (244, 276, 8, "Monster passes within one cell at step 258; the episode survives. Gold rings: evaluator overlay."),
        (276, 401, 4, "Near-median validation example / exploration continues."),
        (401, 909, 2, "Turning loop / fast-forward"),
        (909, 1180, 10, "Exploration resumes; final coverage reaches 89.9%, with all 5 POIs."),
        (1180, 1536, 2, "Turning in place until the time limit / fast-forward"),
    ]
    failure_phases = [
        (0, 128, 2, "Same checkpoint, Hard map 10000: early exploration."),
        (128, 793, 3, "Turning loop / 665 actions / fast-forward"),
        (793, 1536, 1, "Later movement and another loop / fast-forward"),
    ]
    try:
        still("Exploring an unknown world", slide("Explore, remember, and discover", [
            "The agent explores rooms and corridors from a local 7 x 7 view.",
            "It remembers observed terrain and searches for green points of interest.",
            "A moving red patrol makes exploration a survival task as well.",
            "PPO controls turns, forward movement, and waiting."],
            "Built on Farama Foundation MiniGrid (Apache-2.0)"), 12)
        clip("Interactive inspection: seed, difficulty, and agent", ui_video, 32,
             "Choose a map and compare agents. The ground-truth view is for inspection; policy inputs remain partial.")
        still("Procedural maps at three scales", np.asarray(Image.open(output / "generation.png").convert("RGB")), 12)
        clip("Easy / building a map from local observations", output / "replays/aligned-easy/episode.mp4", 24,
             note="The map grows as the agent observes new cells and discovers POIs.")
        clip("Medium / navigating with spatial history", output / "replays/aligned-medium/episode.mp4", 28,
             note="Discovered POIs stay in memory; threat positions are shown only while visible.")
        clip("Hard / near-median validation episode", output / "replays/aligned-hard-typical/episode.mp4", 36,
             phases=hard_phases, highlight=highlight)
        clip("Frontier / BFS toward unexplored space", output / "replays/frontier-medium/episode.mp4", 18,
             note="The baseline searches the same observed map; it has no access to hidden terrain.")
        clip("Threat contact ends the episode", output / "replays/aligned-contact/episode.mp4", 18,
             note="Exploration progress does not guarantee survival.")
        still("Comparing exploration, discovery, and survival", np.asarray(Image.open(output / "comparison.png").convert("RGB")), 22)
        still("The contribution of exploration rewards", slide("Exploration rewards improve coverage", [
            "With rewards for newly seen and visited cells: 90.8% coverage.",
            "Without those rewards: 59.6%, using the same aligned map and PPO settings.",
            "Both conditions retain POI rewards, action costs, and collision penalties.",
            "The comparison uses three training seeds and the same reserved maps."],
            "Means across Easy, Medium, and Hard. Full results and uncertainty are in the results archive."), 14)
        clip("Failure / the same policy can enter a turning loop", output / "replays/aligned-hard/episode.mp4", 8,
             phases=failure_phases)
        still("Where the policy still struggles", slide("Limits of the learned behavior", [
            "Hard maps can produce turning loops and incomplete exploration.",
            "Threats outside the current view are not tracked in memory.",
            "Better coverage does not always mean better survival."],
            "The replay examples illustrate behavior; aggregate performance comes from the full evaluation."), 8)
        still("Run the saved policy", slide("Try the agent on Windows", [
            "Extract the ZIP and open Launch UI.bat.",
            "Choose a seed and difficulty; run, pause, or single-step.",
            "CPU inference: no Python installation, GPU, or retraining needed."],
            "github.com/emkienngu123/MiniGridRL | Farama Foundation MiniGrid (Apache-2.0)"), 8)
    finally:
        writer.close()
    assert elapsed == 240, elapsed
    movie = imageio.get_reader(pending_video)
    try:
        count = movie.count_frames()
        fps = movie.get_meta_data()["fps"]
        assert count == 240 * FPS and count / fps == 240
        assert movie.get_data(count - 1).shape == (720, 1280, 3)
    finally:
        movie.close()
    pending_video.replace(output / "demo.mp4")
    for obsolete in (output / "previews").glob("*.png"):
        parts = obsolete.stem.split("-")
        generated_name = 1 <= len(parts) <= 2 and all(part.isdecimal() for part in parts)
        if generated_name and obsolete.name not in preview_names:
            assert obsolete.resolve().is_relative_to(ROOT)
            obsolete.unlink()
    manifest = {**metadata, "demo_seconds": 240, "fps": FPS, "frames": count,
                "video_sha256": digest(output / "demo.mp4"), "ui_source_sha256": digest(ui_video),
                "storyboard": storyboard, "preview_files": sorted(preview_names), "audio": "Silent; explanatory on-screen captions"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "storyboard.json").write_text(json.dumps(storyboard, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"video": str(output / "demo.mp4"), "seconds": elapsed, "sha256": manifest["video_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission/final")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--ui-video")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if ROOT not in output.parents or output in {ROOT / "submission", ROOT / "submission/aligned"}:
        raise ValueError("Use a new submission subdirectory; historical artifacts must remain intact")
    prepare(output)
    if not args.prepare_only:
        if not args.ui_video or not Path(args.ui_video).is_file():
            raise ValueError("Supply an actual packaged-UI recording with --ui-video")
        render(output, Path(args.ui_video).resolve())


if __name__ == "__main__":
    main()
