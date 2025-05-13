import argparse
import subprocess
from multiprocessing import Pool
from pathlib import Path

AXES_PRESETS_24 = {
    "A6": [0.342, 0.68399, 1.02599]
}
ANGLE_PRESETS = {
    "R5": [80, 45, 15]
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
    parser.add_argument("--n-workers", type=int, default=4)
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
                    project="Target M Sweep",
                    name=f"{a_id}-{r_id}-m{m}-t{trial:02d}",
                    axes=axes,
                    angles=angles,
                    trial=trial,
                    target_m=m,
                    target_radius=0.6203504909,
                    alpha=1,
                    vis_enabled=int(trial < 2)
                )
                jobs.append(cmd)

    for n_sample in [500, 1000, 2000, 10000]:
        for trial in range(10):
            cmd = make_args(
                problem="chamferboundary",
                project="N Sample Sweep",
                name=f"{a_id}-{r_id}-ns{n_sample}-t{trial:02d}",
                axes=axes,
                angles=angles,
                trial=trial,
                m_sample=n_sample,
                target_m=500,
                target_radius=0.6203504909,
                alpha=1,
                vis_enabled=int(trial < 2)
            )
            jobs.append(cmd)

    with Pool(processes=args.n_workers) as pool:
        pool.map(run_process, jobs)

if __name__ == "__main__":
    cli()
