import argparse
import subprocess
from multiprocessing import Pool

def make_args(**kwargs):
    args = ["python", "-m", "framework_mesh.worker"]
    for k, v in kwargs.items():
        if isinstance(v, list):
            args += [f"--{k}"] + [str(x) for x in v]
        else:
            args += [f"--{k}", str(v)]
    return args

def run_process(cmd):
    subprocess.run(cmd)

def sweep_alpha(base_cfg, alphas, n_trials=1):
    jobs = []
    for alpha in alphas:
        for trial in range(n_trials):
            cfg = base_cfg.copy()
            cfg.update({
                "alpha": alpha,
                "trial": trial,
                "name": f"{cfg['object_name']}_a{alpha}_t{trial:02d}"
            })
            jobs.append(make_args(**cfg))
    return jobs

def sweep_objects(base_cfg, object_list, n_trials=1):
    jobs = []
    for obj in object_list:
        for trial in range(n_trials):
            cfg = base_cfg.copy()
            cfg.update({
                "object_name": obj,
                "trial": trial,
                "name": f"{obj}_t{trial:02d}"
            })
            jobs.append(make_args(**cfg))
    return jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()

    base_cfg = {
        "problem": "chamfersampled",
        "project": "Alpha Sweep",
        "object_name": "Balloon0",
        "lr": 1e-4,
        "m_sample": 500,
        "target_m": 100,
        "target_radius": 0.62,
        "vis_enabled": 1
    }

    # Choose one of the sweeps:
    jobs = sweep_alpha(base_cfg, alphas=[0.01, 0.02, 0.05], n_trials=3)
    # jobs = sweep_objects(base_cfg, object_list=["Balloon0", "Balloon1"], n_trials=2)

    print(f"Launching {len(jobs)} jobs using {args.n_workers} workers...")
    with Pool(processes=args.n_workers) as pool:
        pool.map(run_process, jobs)
