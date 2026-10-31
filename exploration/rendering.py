from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size=20, bold=False):
    candidates = ["C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
                  "DejaVuSans.ttf"]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def memory_image(memory, size=440):
    pixels = memory.render(scale=1)
    known = np.argwhere(memory.map[0] > 0)
    if len(known):
        lo = np.maximum(known.min(axis=0) - 3, 0)
        hi = np.minimum(known.max(axis=0) + 4, pixels.shape[:2])
        pixels = pixels[lo[0]:hi[0], lo[1]:hi[1]]
    image = Image.fromarray(pixels)
    image.thumbnail((size, size), Image.Resampling.NEAREST)
    # thumbnail never upscales, so explicitly scale the observed crop.
    factor = size / max(image.size)
    image = image.resize((int(image.width * factor), int(image.height * factor)), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size), "#121928")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def episode_frame(env, title="MiniGrid exploration", subtitle=""):
    from exploration.evaluation import episode_metrics

    canvas = Image.new("RGB", (1280, 720), "#0b1120")
    draw = ImageDraw.Draw(canvas)
    draw.text((38, 25), title, font=font(30, True), fill="#f2f5fb")
    draw.text((38, 68), subtitle, font=font(17), fill="#93a4bd")
    draw.text((38, 108), "EVALUATOR VIEW / ground truth", font=font(17, True), fill="#efbe76")
    memory_title = "HISTORY TRACKER / excluded from policy" if env.variant == "no_memory" else "AGENT MEMORY / observed cells only"
    draw.text((677, 108), memory_title, font=font(17, True), fill="#74d4c1")
    view = Image.fromarray(env.unwrapped.get_frame(tile_size=20))
    view.thumbnail((540, 495), Image.Resampling.NEAREST)
    canvas.paste(view, (38 + (540 - view.width) // 2, 140 + (495 - view.height) // 2))
    memory = memory_image(env.memory, 495)
    canvas.paste(memory, (683, 140))
    metrics = episode_metrics(env)
    text = (f"Step {metrics['steps']:4d}/{metrics['episode_limit']}     "
            f"Coverage {metrics['revealed_coverage']:.1%}     "
            f"POIs {metrics['poi_discovered']}/{env.unwrapped.poi_count}     "
            f"Wall contacts {metrics['wall_collisions']}     Monster contacts {metrics['monster_collisions']}")
    draw.text((38, 666), text, font=font(21), fill="#f2f5fb")
    return np.asarray(canvas)
