import json
import argparse
from copy import deepcopy
from .experiments import EllipsoidExperiment
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path



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
    "training": {"n_iters": 200, "lr": 1e-2, "momentum": 0.7},
    "verbose": False,
    "vis": {"enabled": True, "frequency": 1, "backend": "plotly"},
    "wandb": True,
}

# ----------------------------- MAIN LOGIC --------------------------------- #

def run_all(outdir, save_cfgs: bool = False):
    outdir.mkdir(parents=True, exist_ok=True)

    for i, (problem, prod) in enumerate(PROBLEM_SETTINGS):
        axes_presets = AXES_PRESETS_02 if prod == 0.02 else AXES_PRESETS_24
        angle_presets = {"R1": ANGLE_PRESETS["R1"]} if problem == "axisaligned" else ANGLE_PRESETS

        for a_id, axes in axes_presets.items():
            for r_id, angles in angle_presets.items():
                for trial in range(10):
                    cfg = json.loads(json.dumps(BASE_CFG))  # deep copy
                    cfg.update({
                        "problem": problem,
                        "name": f"{problem}-{a_id}-{r_id}-t{trial:02d}",
                        "initial_axes": axes,
                        "initial_angles": angles,
                        "axes_id": a_id,          # <-- add this
                        "rotation_id": r_id,      # <-- add this
                        "trial": trial,            # <-- add this
                        "vis_enabled": trial < 2,
                    })

                    # Adjust learning rate based on problem index
                    cfg["training"]["lr"] = 1e-2 if i < 2 else 5e-1
                    cfg["target"]["radius"] = 0.2820947918 if i < 2 else 0.6203504909

                    # Per‑problem tweaks
                    if problem in {"chamfersampled", "chamferboundary"}:
                        cfg["view_indices"] = [0, 2, 4, 6]
                    
                    # vis frequency & switch off after 2 runs
                    cfg["vis"]["enabled"] = trial < 2


                    # ----------------- RUN EXPERIMENT -------------------- #
                    experiment = EllipsoidExperiment(cfg)
                    experiment.run()


def cli():
    p = argparse.ArgumentParser(description="Batch runner for ellipsoid problems")
    p.add_argument("--outdir", type=Path, default="runs", help="Directory for configs & logs")
    p.add_argument("--no-save-cfgs", action="store_true", help="Do not write config files to disk")
    args = p.parse_args()
    run_all(args.outdir, save_cfgs=not args.no_save_cfgs)


if __name__ == "__main__":
    cli()
