import json
import argparse
from copy import deepcopy
from .experiments import EllipsoidExperiment
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import random

from .experiments import EllipsoidExperiment  # adjust import if located elsewhere

# ----------------------------- PRESETS ------------------------------------ #
AXES_PRESETS_02 = {
    "A1": [0.271, 0.271, 0.271],            # sphere
    "A2": [0.215, 0.215, 0.431],            # oblate spheroid
    "A3": [0.171, 0.342, 0.342],            # prolate spheroid
    "A4": [0.22314, 0.26777, 0.33472],            # triaxial 1:1.2:1.5
    "A5": [0.18821, 0.28231, 0.37641],            # triaxial 1:1.5:2
    "A6": [0.14938, 0.29876, 0.44814]
}

AXES_PRESETS_24 = {
    "A1": [0.620, 0.620, 0.620],
    "A2": [0.384, 0.384, 0.768],
    "A3": [0.247, 0.494, 0.494],
    "A4": [0.51087, 0.61305, 0.76631],
    "A5": [0.43089, 0.64633, 0.86177],
    "A6": [0.342, 0.68399, 1.02599]
}

ANGLE_PRESETS = {
    "R1": [0, 0, 0],
    "R2": [30, 30, 0],
    "R3": [45, 45, 45],
    "R4": [45, 60, 80],
    "R5": [80, 45, 15],
}

PROBLEM_SETTINGS = [
    ("axisaligned", 0.02),
    ("nonaxisaligned", 0.02),
    ("chamfersampled", 0.24),
    ("chamferboundary", 0.24),
    ("volumeconstrained", 0.24),
]

# Base configuration shared by all runs
BASE_CFG = {
    "project": "grid_init_uniform_sampling",
    "name": "auto-run",
    "sqrt_m": 30,
    "seed": 42,
    "noise": 1e-2,
    "p": 1.6075,
    "initial_axes": [0.2, 0.2, 0.2],  # placeholder, will be overwritten
    "initial_angles": [0, 0, 0],       # placeholder, will be overwritten
    "target": {"radius": 0.2820947918, "m": 300, "sqrt_m": 30},
    "view_mode": "angles",
    "view_angles": [
        [-90.0, -90.0, 0.0],
        [180.0, 0.0, 90.0],
        [90.0, 0.0, 0.0],
        [-150.0, -35.2643897, 45.0],
        [-30.0, -35.2643897, -45.0],
        [150.0, 35.2643897, 45.0],
        [30.0, 35.2643897, -45.0],
        [150.0, 35.2643897, 45.0],
    ],
    "view_normals": [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, -1, 1],
        [-1, 1, 1], [-1, -1, 1], [-1, 1, 1],
    ],
    "view_indices": [0, 2, 4, 6],  # will be overridden for chamfer problems
    "training": {"n_iters": 200, "lr": 1e-2, "momentum": 0.8},
    "verbose": False,
    "vis": {"enabled": True, "frequency": 1, "backend": "plotly"},
    "wandb": True,
}

# ----------------------------- MAIN LOGIC --------------------------------- #
def run_noise_sweep(outdir, noise_levels, save_cfgs=False):
    outdir.mkdir(parents=True, exist_ok=True)
    configs = [("A1", "R1"), ("A6", "R5")]
    problems = ["volumeconstrained", "chamfersampled", "chamferboundary"]

    # Point to resume from:
    resume_from = {
        "problem": "chamfersampled",
        "axes_id": "A6",
        "rotation_id": "R5",
        "noise": 0.0,
        "trial": 0,
    }
    resuming = False

    for problem in problems:
        for a_id, r_id in configs:
            axes = AXES_PRESETS_24[a_id]
            angles = ANGLE_PRESETS[r_id]
            for noise in noise_levels:
                for trial in range(10):
                    if not resuming:
                        if (
                            problem == resume_from["problem"]
                            and a_id == resume_from["axes_id"]
                            and r_id == resume_from["rotation_id"]
                            and noise == resume_from["noise"]
                            and trial == resume_from["trial"]
                        ):
                            resuming = True  # start running from this point
                        else:
                            continue  # skip before resume point

                    cfg = json.loads(json.dumps(BASE_CFG))
                    cfg.update({
                        "problem": problem,
                        "name": f"{problem}-{a_id}-{r_id}-n{noise:.0e}-t{trial:02d}",
                        "initial_axes": axes,
                        "initial_angles": angles,
                        "axes_id": a_id,
                        "rotation_id": r_id,
                        "trial": trial,
                        "noise": noise,
                        "vis_enabled": trial < 2,
                    })
                    cfg["project"] = "Noise Sweep"
                    cfg["training"]["lr"] = 5e-1
                    cfg["target"]["radius"] = 0.6203504909
                    cfg["view_indices"] = [0, 2, 4, 6]
                    cfg["vis"]["enabled"] = trial < 2
                    experiment = EllipsoidExperiment(cfg)
                    experiment.run()


def run_alpha_sweep(outdir, alpha_values, save_cfgs=False):
    outdir.mkdir(parents=True, exist_ok=True)
    configs = [("A6", "R5")]
    problems = ["chamferboundary"]

    for problem in problems:
        for a_id, r_id in configs:
            axes = AXES_PRESETS_24[a_id]
            angles = ANGLE_PRESETS[r_id]
            for alpha in alpha_values:
                for trial in range(5):
                    cfg = json.loads(json.dumps(BASE_CFG))
                    cfg.update({
                        "problem": problem,
                        "name": f"{problem}-{a_id}-{r_id}-a{alpha}-t{trial:02d}",
                        "initial_axes": axes,
                        "initial_angles": angles,
                        "axes_id": a_id,
                        "rotation_id": r_id,
                        "trial": trial,
                        "alpha": alpha,
                        "noise": 1e-2,
                        "vis_enabled": trial < 2,
                    })
                    cfg["project"] = "Alpha Sweep"
                    cfg["training"]["lr"] = 5e-1
                    cfg["target"]["radius"] = 0.6203504909
                    cfg["view_indices"] = [0, 2, 4, 6]
                    cfg["vis"]["enabled"] = trial < 2
                    experiment = EllipsoidExperiment(cfg)
                    experiment.run()


def run_view_count_sweep(outdir, view_counts, save_cfgs=False):
    outdir.mkdir(parents=True, exist_ok=True)
    configs = [("A6", "R5")]
    problems = ["chamfersampled", "chamferboundary"]
    full_view_indices = list(range(len(BASE_CFG["view_angles"])))

    for problem in problems:
        for a_id, r_id in configs:
            axes = AXES_PRESETS_24[a_id]
            angles = ANGLE_PRESETS[r_id]
            for n_views in view_counts:
                for trial in range(10):
                    selected_views = sorted(random.sample(full_view_indices, k=n_views))

                    cfg = json.loads(json.dumps(BASE_CFG))
                    cfg.update({
                        "problem": problem,
                        "project": "Num Views Sweep",
                        "name": f"{a_id}-{r_id}-v{n_views}-t{trial:02d}",
                        "initial_axes": axes,
                        "initial_angles": angles,
                        "axes_id": a_id,
                        "rotation_id": r_id,
                        "trial": trial,
                        "view_indices": selected_views,
                        "num_views": n_views,
                        "noise": 1e-2,
                        "alpha": 5.0,
                        "vis_enabled": trial < 2,
                    })

                    cfg["training"]["lr"] = 5e-1
                    cfg["target"]["radius"] = 0.6203504909
                    cfg["vis"]["enabled"] = trial < 2
                    if problem == "chamfersampled":
                        cfg["training"]["n_iters"] = 300 if n_views in {1, 2, 3} else 200
                    elif problem == "chamferboundary":
                        cfg["training"]["n_iters"] = 400 if n_views in {1, 2, 3} else 200

                    if save_cfgs:
                        config_path = outdir / f"{cfg['name']}.json"
                        with config_path.open("w") as f:
                            json.dump(cfg, f, indent=2)

                    experiment = EllipsoidExperiment(cfg)
                    experiment.run()


def cli():
    p = argparse.ArgumentParser(description="Batch runner for ellipsoid problems")
    p.add_argument("--outdir", type=Path, default="runs", help="Directory for configs & logs")
    p.add_argument("--no-save-cfgs", action="store_true", help="Do not write config files to disk")
    args = p.parse_args()

    # run_alpha_sweep(args.outdir, alpha_values=[0, 10, 20, 30, 50])
    run_view_count_sweep(args.outdir, view_counts=[1, 2, 3, 4, 6, 8])

if __name__ == "__main__":
    cli()
