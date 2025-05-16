import argparse
import subprocess
from multiprocessing import Pool
from pathlib import Path

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
BASE_CFG = {
    "training": {"lr": 5e-1},
    "target": {"radius": 0.6203504909},
    "view_indices": [0, 2, 4, 6],
    "vis": {"enabled": True},
    "vis_enabled": True
}

def make_args(**kwargs):
    args = ["python", "-m", "framework_ellipsoid.worker"]
    for k, v in kwargs.items():
        if isinstance(v, list):
            args += [f"--{k}"] + [str(x) for x in v]
        else:
            args += [f"--{k}", str(v)]
    return args

def run_process(cmd):
    subprocess.run(cmd)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=5)
    args = parser.parse_args()

    a_id, r_id = "A6", "R5"
    axes = AXES_PRESETS_24[a_id]
    angles = ANGLE_PRESETS[r_id]

    jobs = []

    for problem in ["chamfersampled", "chamferboundary"]:
        for m in [50, 250, 500, 1000]:
            for trial in range(10):
                cmd = make_args(
                    problem=problem,
                    project="Target M Sweep2",
                    name=f"{a_id}-{r_id}-m{m}-t{trial:02d}",
                    axes=axes,
                    angles=angles,
                    trial=trial,
                    target_m=m,
                    m_sample=500,
                    target_radius=0.6203504909,
                    alpha=1,
                    vis_enabled=int(trial < 2)
                )
                jobs.append(cmd)

    # for problem in ["chamfersampled", "chamferboundary"]:
    #     for n_sample in [200, 500, 1000, 2000, 5000]:
    #         for trial in range(10):
    #             cmd = make_args(
    #                 problem=problem,
    #                 project="Trial",
    #                 name=f"{a_id}-{r_id}-m{m}-t{trial:02d}",
    #                 axes=axes,
    #                 angles=angles,
    #                 trial=trial,
    #                 m_sample=n_sample,
    #                 target_m=500,
    #                 target_radius=0.6203504909,
    #                 alpha=0,
    #                 n_iters=300,
    #                 lr=5e-1,
    #                 vis_enabled=int(trial < 2)
    #             )
    #             jobs.append(cmd)

    for problem in ["chamfersampled"]:
        for msample in [55]:
            for trial in range(1):
                cmd = make_args(
                    problem=problem,
                    project="Trial 4",
                    name=f"{a_id}-{r_id}-msample{msample}-t{trial:02d}",
                    axes=axes,
                    angles=angles,
                    trial=trial,
                    target_m=300,
                    m_sample=msample,
                    target_radius=0.6203504909,
                    alpha=0,
                    noise=1e-2,
                    vis_enabled= int(trial<1)
                )
                jobs.append(cmd)

    # for alpha in [0]:
    #     for problem in ["chamferboundary"]:
    #         for msample in [200,500,1000,2000,5000]:
    #             for trial in range(10):
    #                 cmd = make_args(
    #                     problem=problem,
    #                     project="Trial 4",
    #                     name=f"{a_id}-{r_id}-msample{msample}-t{trial:02d}",
    #                     axes=axes,
    #                     angles=angles,
    #                     trial=trial,
    #                     target_m=100,
    #                     m_sample=msample,
    #                     target_radius=0.6203504909,
    #                     alpha=alpha,
    #                     noise=1e-2,
    #                     vis_enabled= int(trial<1)
    #                 )
    #                 jobs.append(cmd)



    # for problem in ["chamfersampled", "chamferboundary"]:
    #     for n_sample in [200, 500, 1000, 2000, 5000]:
    #         for trial in range(10):
    #             cmd = make_args(
    #                 problem=problem,
    #                 project="N Sample Sweep2",
    #                 name=f"{a_id}-{r_id}-ns{n_sample}-t{trial:02d}",
    #                 axes=axes,
    #                 angles=angles,
    #                 trial=trial,
    #                 m_sample=n_sample,
    #                 target_m=500,
    #                 target_radius=0.6203504909,
    #                 alpha=0,
    #                 n_iters=300,
    #                 lr=5e-1,
    #                 vis_enabled=int(trial < 2)
    #             )
    #             jobs.append(cmd)

    with Pool(processes=args.n_workers) as pool:
        pool.map(run_process, jobs)

# [5e-2, 5e-1, 1]
def run_noise_sweep_parallel(n_workers=4, noise_levels=[0.001, 0.01, 0.05, 0.1, 0.5]):
    configs = [("A6", "R5")]
    problems = ["chamferboundary"]

    # Resume point
    resume_from = {
        "problem": "chamfersampled",
        "axes_id": "A6",
        "rotation_id": "R5",
        "noise": 0.0,
        "trial": 0,
    }
    resuming = True

    jobs = []

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
                            resuming = True
                        else:
                            continue  # Skip until resume point

                    name = f"{a_id}-{r_id}-n{noise:.0e}-t{trial:02d}"
                    cfg = {
                        "problem": problem,
                        "project": "Noise Sweep",
                        "name": name,
                        "axes": axes,
                        "angles": angles,
                        "axes_id": a_id,
                        "rotation_id": r_id,
                        "trial": trial,
                        "noise": noise,
                        "lr": 5e-1,
                        "target_radius": BASE_CFG["target"]["radius"],
                        "view_indices": BASE_CFG["view_indices"],
                        "vis_enabled": int(trial < 2)
                    }
                    cmd = make_args(**cfg)
                    jobs.append(cmd)

    print(f"Launching {len(jobs)} jobs using {n_workers} workers...")

    with Pool(processes=n_workers) as pool:
        pool.map(run_process, jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=6)
    args = parser.parse_args()
    cli()
