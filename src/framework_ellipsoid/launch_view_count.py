# launch_view_count_sweep.py
import subprocess
import random
from multiprocessing import Pool

AXES_PRESETS_24 = {
    "A6": [0.342, 0.68399, 1.02599]
}
ANGLE_PRESETS = {
    "R5": [80, 45, 15]
}
FULL_VIEW_INDICES = list(range(8))

def make_args(**kwargs):
    args = ["python", "-m", "framework_ellipsoid.worker"]
    for k, v in kwargs.items():
        if isinstance(v, list):
            args += [f"--{k}"] + [str(x) for x in v]
        else:
            args += [f"--{k}", str(v)]
    return args

def run_process(cmd):
    subprocess.run(cmd, check=True)

def main():
    a_id, r_id = "A6", "R5"
    axes = AXES_PRESETS_24[a_id]
    angles = ANGLE_PRESETS[r_id]
    problems = ["chamfersampled", "chamferboundary"]
    view_counts = [1, 2, 3, 4, 6, 8]
    n_trials = 10
    n_workers = 4

    jobs = []

    for problem in problems:
        for n_views in view_counts:
            for trial in range(n_trials):
                selected_views = sorted(random.sample(FULL_VIEW_INDICES, k=n_views))

                cmd = make_args(
                    problem=problem,
                    project="Num Views Sweep",
                    name=f"{problem}-{a_id}-{r_id}-v{n_views}-t{trial:02d}",
                    axes=axes,
                    angles=angles,
                    trial=trial,
                    view_indices=selected_views,
                    num_views=n_views,
                    noise=1e-2,
                    alpha=1.0,
                    vis_enabled="1" if trial < 2 else "0",
                    n_iters=300 if problem == "chamfersampled" else 400,
                    lr=5e-1,
                    target_radius=0.6203504909
                )

                jobs.append(cmd)

    with Pool(processes=n_workers) as pool:
        pool.map(run_process, jobs)

if __name__ == "__main__":
    main()
