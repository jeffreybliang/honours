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
    "A4": [0.150, 0.300, 0.450],            # triaxial 1:2:3
    "A5": [0.163, 0.245, 0.490],            # triaxial 1:1.5:3
}

AXES_PRESETS_24 = {
    "A1": [0.620, 0.620, 0.620],
    "A2": [0.384, 0.384, 0.768],
    "A3": [0.247, 0.494, 0.494],
    "A4": [0.223, 0.446, 0.669],
    "A5": [0.254, 0.381, 0.763],
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
    "project": "ellipsoid-experiments",
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
        if problem not in {"chamferboundary", "volumeconstrained"}:
            continue  # skip all other problems

        axes_presets = AXES_PRESETS_02 if prod == 0.02 else AXES_PRESETS_24
        angle_presets = {"R1": ANGLE_PRESETS["R1"]} if problem == "axisaligned" else ANGLE_PRESETS

        for a_id, axes in axes_presets.items():
            # Restrict A3–A5 for chamferboundary, A1–A5 for volumeconstrained
            if problem == "chamferboundary" and a_id not in {"A3", "A4", "A5"}:
                continue
            if problem == "volumeconstrained" and a_id not in {"A1", "A2", "A3", "A4", "A5"}:
                continue

            for r_id, angles in angle_presets.items():
                for trial in range(10):
                    cfg = json.loads(json.dumps(BASE_CFG))  # deep copy
                    cfg.update({
                        "problem": problem,
                        "name": f"{problem}-{a_id}-{r_id}-t{trial:02d}",
                        "initial_axes": axes,
                        "initial_angles": angles,
                        "axes_id": a_id,
                        "rotation_id": r_id,
                        "trial": trial
                    })

                    # Learning rate / radius based on early vs late index
                    cfg["training"]["lr"] = 1e-2 if i < 2 else 2e-1
                    cfg["target"]["radius"] = 0.2820947918 if i < 2 else 0.6203504909

                    if problem in {"chamfersampled", "chamferboundary"}:
                        cfg["view_indices"] = [0, 2, 4, 6]

                    cfg["vis"]["enabled"] = trial < 5

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
