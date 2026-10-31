from __future__ import annotations

import secrets
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from exploration.agents import create_agent
from exploration.config import PRESETS
from exploration.evaluation import episode_metrics
from exploration.memory import make_env
from exploration.rendering import memory_image

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


class ExplorerUI:
    def __init__(self, root, checkpoint=None):
        self.root = root
        self.running = False
        self.env = None
        self.obs = None
        self.agent = None
        self.timer = None
        self.ended = False
        root.title("MiniGrid Explorer | RL inspection studio")
        root.geometry("1210x790")
        root.minsize(1120, 730)
        root.configure(bg="#0b1120")
        self.seed = tk.StringVar(value="10000")
        self.preset = tk.StringVar(value="easy")
        default = ROOT / "checkpoints" / "demo.zip"
        self.checkpoint = tk.StringVar(value=str(checkpoint or (default if default.exists() else "")))
        self.agent_name = tk.StringVar(value="ppo" if self.checkpoint.get() else "frontier")
        self.speed = tk.IntVar(value=12)
        self.size = tk.StringVar()
        self.pois = tk.StringVar()
        self.limit = tk.StringVar()
        self.interval = tk.StringVar()
        self.status = tk.StringVar(value="Ready")
        self.metrics_text = tk.StringVar()
        self._build()
        self.apply_preset()
        self.reset()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#0b1120")
        style.configure("Card.TFrame", background="#131e30")
        style.configure("TLabel", background="#0b1120", foreground="#dbe5f3", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground="#93a4bd")
        style.configure("Title.TLabel", font=("Segoe UI", 23, "bold"), foreground="#f7f9fc")
        style.configure("TButton", font=("Segoe UI", 10), padding=(9, 7))
        style.configure("Accent.TButton", background="#59cbb5", foreground="#091b1a")
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=5)
        head = ttk.Frame(self.root, padding=(24, 20))
        head.pack(fill="x")
        ttk.Label(head, text="MINIGRID / EXPLORER", style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, text="Explore the unknown. Inspect what the policy actually sees.", style="Muted.TLabel").pack(anchor="w", pady=(5, 0))
        body = ttk.Frame(self.root, padding=(24, 0, 24, 18))
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, width=220)
        controls.pack(side="left", fill="y", padx=(0, 22))
        controls.pack_propagate(False)

        def label(text):
            ttk.Label(controls, text=text, style="Muted.TLabel").pack(anchor="w", pady=(10, 4))

        label("MAP SEED")
        seed_row = ttk.Frame(controls)
        seed_row.pack(fill="x")
        ttk.Entry(seed_row, textvariable=self.seed, width=13).pack(side="left", fill="x", expand=True)
        ttk.Button(seed_row, text="Random", command=self.randomize).pack(side="right", padx=(5, 0))
        label("DIFFICULTY")
        box = ttk.Combobox(controls, textvariable=self.preset, values=list(PRESETS), state="readonly")
        box.pack(fill="x")
        box.bind("<<ComboboxSelected>>", lambda _: self.apply_preset())
        grid = ttk.Frame(controls)
        grid.pack(fill="x", pady=(10, 0))
        for row, (text, variable) in enumerate((("Grid size (odd)", self.size), ("POIs", self.pois),
                                                ("Action limit", self.limit), ("Threat interval", self.interval))):
            ttk.Label(grid, text=text).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(grid, textvariable=variable, width=7).grid(row=row, column=1, sticky="e", padx=(8, 0))
        label("AGENT")
        ttk.Combobox(controls, textvariable=self.agent_name, values=["frontier", "random", "ppo"], state="readonly").pack(fill="x")
        ttk.Button(controls, text="Choose checkpoint...", command=self.browse).pack(fill="x", pady=(7, 0))
        self.checkpoint_label = ttk.Label(controls, text="", wraplength=210, style="Muted.TLabel")
        self.checkpoint_label.pack(anchor="w", pady=4)
        ttk.Button(controls, text="Generate / reset map", command=self.reset).pack(fill="x", pady=(10, 5))
        row = ttk.Frame(controls)
        row.pack(fill="x")
        ttk.Button(row, text="Start", style="Accent.TButton", command=self.start).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Pause", command=self.pause).pack(side="left", expand=True, fill="x", padx=(6, 0))
        ttk.Button(controls, text="Single step", command=self.single_step).pack(fill="x", pady=(7, 0))
        label("PLAYBACK SPEED")
        tk.Scale(controls, from_=1, to=60, orient="horizontal", variable=self.speed,
                 bg="#0b1120", fg="#dbe5f3", highlightthickness=0, troughcolor="#24334c").pack(fill="x")
        content = ttk.Frame(body)
        content.pack(side="left", fill="both", expand=True)
        views = ttk.Frame(content)
        views.pack(fill="both", expand=True)
        views.columnconfigure((0, 1), weight=1)
        ttk.Label(views, text="EVALUATOR VIEW", foreground="#efbe76", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(views, text="AGENT MEMORY", foreground="#74d4c1", font=("Segoe UI", 12, "bold")).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(views, text="Ground truth / debugging only", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 10))
        self.memory_mode_label = ttk.Label(views, text="Built only from observations", style="Muted.TLabel")
        self.memory_mode_label.grid(row=1, column=1, sticky="w", padx=(12, 0), pady=(4, 10))
        self.map_label = ttk.Label(views)
        self.map_label.grid(row=2, column=0, sticky="n")
        self.memory_label = ttk.Label(views)
        self.memory_label.grid(row=2, column=1, sticky="n", padx=(12, 0))
        ttk.Label(content, textvariable=self.metrics_text, font=("Consolas", 11), justify="left").pack(anchor="w", pady=(18, 10))
        ttk.Label(content, text="LEGEND   Green: POI   Red disc: monster   Gold: agent in memory", style="Muted.TLabel").pack(anchor="w", pady=(5, 0))
        ttk.Label(content, textvariable=self.status, foreground="#74d4c1", wraplength=850).pack(anchor="w", pady=(12, 0))

    def apply_preset(self):
        settings = PRESETS[self.preset.get()]
        for variable, key in ((self.size, "size"), (self.pois, "poi_count"), (self.limit, "max_steps"), (self.interval, "monster_interval")):
            variable.set(str(settings[key]))

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("PPO checkpoint", "*.zip")])
        if path:
            self.checkpoint.set(path)
            self.agent_name.set("ppo")
            self.checkpoint_label.configure(text=Path(path).name)

    def randomize(self):
        self.seed.set(str(secrets.randbelow(2**31)))
        self.reset()

    def reset(self):
        self.pause()
        try:
            seed = int(self.seed.get())
            if seed < 0:
                raise ValueError("Seed must be nonnegative")
            kwargs = {"size": int(self.size.get()), "poi_count": int(self.pois.get()),
                      "max_steps": int(self.limit.get()), "monster_interval": int(self.interval.get())}
            # Validate/load before replacing a working session.
            agent = create_agent(self.agent_name.get(), self.checkpoint.get() or None)
            metadata = getattr(agent, "metadata", {})
            env = make_env(self.preset.get(), variant=metadata.get("variant", "main"),
                           reward=metadata.get("config", {}).get("reward"), **kwargs)
            obs, _ = env.reset(seed=seed)
            agent.reset(seed)
            if self.env:
                self.env.close()
            self.env, self.obs, self.agent = env, obs, agent
            self.memory_mode_label.configure(text="History tracker / policy memory disabled" if env.variant == "no_memory" else "Built only from observations")
            self.ended = False
            self.checkpoint_label.configure(text=Path(self.checkpoint.get()).name if self.checkpoint.get() else "No checkpoint selected")
            self.status.set(f"Ready / {self.agent.name} / seed {seed}")
            self.draw()
        except (ValueError, OSError) as exc:
            self.status.set(f"Configuration error: {exc}")
            messagebox.showerror("Cannot reset", str(exc))

    def draw(self):
        image = Image.fromarray(self.env.unwrapped.get_frame(tile_size=20))
        image = image.resize((420, 420), Image.Resampling.NEAREST)
        self.map_photo = ImageTk.PhotoImage(image)
        self.memory_photo = ImageTk.PhotoImage(memory_image(self.env.memory, 420))
        self.map_label.configure(image=self.map_photo)
        self.memory_label.configure(image=self.memory_photo)
        metrics = episode_metrics(self.env)
        self.metrics_text.set(
            f"STEP {metrics['steps']:4d}/{metrics['episode_limit']:<4d}    COVERAGE {metrics['revealed_coverage']:6.1%}    VISITED {metrics['visited_coverage']:6.1%}\n"
            f"POIs {metrics['poi_discovered']}/{self.env.unwrapped.poi_count}           WALL CONTACTS {metrics['wall_collisions']:<4d}    LOOPS {metrics['loop_rate']:6.1%}"
        )

    def advance(self):
        if self.ended or not self.env:
            return
        action = self.agent.act(self.obs)
        self.obs, _, terminated, truncated, _ = self.env.step(action)
        self.ended = terminated or truncated
        self.draw()
        if self.ended:
            self.pause()
            self.status.set("Episode ended / monster collision" if terminated else "Episode completed / action limit reached")

    def _tick(self):
        self.timer = None
        if self.running:
            self.advance()
            if self.running:
                self.timer = self.root.after(max(1, 1000 // self.speed.get()), self._tick)

    def start(self):
        if not self.running and not self.ended:
            self.running = True
            self.status.set(f"Running / {self.agent.name} / CPU inference")
            self._tick()

    def pause(self):
        self.running = False
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None
        self.status.set("Paused")

    def single_step(self):
        self.pause()
        self.advance()

    def close(self):
        self.pause()
        if self.env:
            self.env.close()
        self.root.destroy()


def launch(checkpoint=None):
    # Keep CPU inference responsive on machines with many logical cores.
    import torch
    torch.set_num_threads(1)
    root = tk.Tk()
    ExplorerUI(root, checkpoint)
    root.mainloop()
