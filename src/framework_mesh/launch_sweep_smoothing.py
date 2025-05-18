import argparse
import subprocess
from multiprocessing import Pool

# --- CONFIG ---
data_path = "./framework_mesh/data_diffuse.json"
exp_path = "./framework_mesh/exp_all_diffuse.json"
device = "cpu"

objects = ["Diamond", "Balloon", "Spiky", "Parabola", "Biconvex"]
alpha_override = {
    "Diamond": 2,
    "Balloon": 2,
    "Spiky": 10,
    "Parabola": 10,
    "Biconvex": 5,
}
projection_modes = ["alpha"]  # singleton for extensibility

def make_args(**kwargs):
    args = ["python", "-m", "framework_mesh.worker"]
    for k, v in kwargs.items():
        args += [f"--{k}", str(v)]
    return args

def run_process(cmd):
    subprocess.run(cmd)

def sweep_gradient_smoothing():
    mesh_res_vals = [2, 3]
    smoothing_methods = ["jacobi", "invhop", "khop"]
    constrained_flags = [False, True]
    smoothing_ks = [1, 3, 5]
    trials = 2
    jobs = []

    for mesh_res in mesh_res_vals:
        for obj in objects:
            alpha = alpha_override[obj]
            ntrials = trials if mesh_res == 2 else 1
            for mode in projection_modes:
                for method in smoothing_methods:
                    for constrained in constrained_flags:
                        k_vals = smoothing_ks if method in {"jacobi", "khop"} else [0]
                        for k in k_vals:
                            # Adjust learning rate and momentum based on k
                            if k == 1:
                                lr, momentum = 5e-6, 0.8
                            elif k == 3:
                                lr, momentum = 5e-6, 0.9
                            elif k == 5:
                                lr, momentum = 8e-6, 0.9
                            else:
                                lr, momentum = 5e-6, 0.85  # default

                            # Adjust number of iterations based on k
                            n_iters = 120 + 10 * max(0, k - 1)

                            for trial in range(ntrials):
                                vis = int(trial == 0 and mesh_res == 2)
                                run_name = f"{obj}_res{mesh_res}_{method}_k{k}_constr{int(constrained)}_t{trial:02d}"

                                cmd = make_args(
                                    data_path=data_path,
                                    exp_base_path=exp_path,
                                    target_object=obj,
                                    projection_mode=mode,
                                    mesh_res=mesh_res,
                                    alpha=alpha,
                                    name=run_name,
                                    device=device,
                                    vis_enabled=vis,
                                    velocity_enabled=0,
                                    smoothing_enabled=1,
                                    smoothing_method=method,
                                    smoothing_k=k,
                                    constrained=constrained,
                                    project=f"Shape Smoothing Sweep Res{mesh_res} v2",
                                    n_iters=n_iters,
                                    lr=lr,
                                    momentum=momentum,
                                    view_mode="manual",
                                    num_views=12,
                                )
                                jobs.append(cmd)
    return jobs


def main(n_workers):
    jobs = sweep_gradient_smoothing()
    print(f"Launching {len(jobs)} jobs with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        pool.map(run_process, jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=3)
    args = parser.parse_args()
    main(args.n_workers)
