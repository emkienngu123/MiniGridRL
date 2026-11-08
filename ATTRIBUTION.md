# Attribution and licenses

This repository extends [Farama Foundation MiniGrid](https://github.com/Farama-Foundation/Minigrid), starting from commit `7f5853792ea7f04f75614661006d87fcf949f26d`. Upstream history, source, copyright notices, [citation metadata](CITATION.cff), and the root [Apache 2.0 LICENSE](LICENSE) are preserved. The original README is retained as `README.minigrid.md`. Assignment additions use the repository's retained license; dependencies retain their own licenses. Earlier project prose incorrectly called MiniGrid MIT; the unchanged upstream LICENSE is authoritative.

The environment subclasses `MiniGridEnv` and uses MiniGrid's Grid, walls, floor objects, ball rendering, visibility processing, observations, lifecycle, and native turn/forward execution. The map generator, observation-only memory, frontier baseline, rewards, metric code, integration scripts, and inspector were written for this assignment. The PPO implementation is provided by Stable-Baselines3 and is not claimed as original work.

| Dependency | Role | License / upstream |
|---|---|---|
| MiniGrid | Environment foundation | Apache-2.0; Farama Foundation |
| Gymnasium | Standard environment API and spaces | MIT; Farama Foundation |
| Stable-Baselines3 | PPO implementation, rollout buffers, training tools | MIT; DLR-RM/stable-baselines3 |
| PyTorch | Neural networks and inference | BSD-style; pytorch/pytorch; bundled dependencies have their own notices |
| NumPy | Arrays and seeded random generators | BSD-3-Clause |
| pygame-ce | MiniGrid rendering | LGPL-2.1-or-later |
| Pillow | UI image presentation and video captions | HPND/PIL license |
| Tcl/Tk | Native Python UI | Tcl/Tk license |
| PyYAML | Configuration | MIT |
| Matplotlib | Experiment plots | Matplotlib license, BSD-compatible |
| TensorBoard | Training logs | Apache-2.0 |
| ImageIO / imageio-ffmpeg | Replay/demo encoding | BSD-2-Clause; FFmpeg binaries carry their own LGPL/GPL notices |
| ReportLab | Technical PDF report | BSD-style |
| pypdf / pypdfium2 | PDF validation/rendering | BSD-3-Clause / Apache-2.0 or BSD-3-Clause; PDFium third-party notices |
| PyInstaller | Standalone build | GPL-2.0-or-later with the bootloader distribution exception |
| pytest / Ruff | Development checks | MIT / MIT |

Pinned direct and transitive dependency versions are recorded in the CPU/CUDA lock files. The portable distribution includes dependency license files collected from the installed CPU runtime, in addition to notices packaged by PyInstaller. FFmpeg is used to generate MP4 files; no FFmpeg executable is required by the evaluator for inference.

Research references:

- Chevalier-Boisvert et al. (2023), *Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks*. See upstream `CITATION.cff`.
- Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Raffin et al. (2021), [Stable-Baselines3: Reliable Reinforcement Learning Implementations](https://jmlr.org/papers/v22/20-1364.html).

No research implementation beyond the attributed libraries was copied into the assignment code.
