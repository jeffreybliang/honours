import argparse
import subprocess
from multiprocessing import Pool

# --- CONFIG ---
data_path = "./framework_mesh/data_diffuse.json"
exp_path = "./framework_mesh/exp_all_diffuse.json"
device = "cpu"

objects = ["Diamond", "Balloon", "Spiky", "Biconvex", "Cylinder"]
# objects = ["Parabola"] 
alpha_override = {
    "Diamond": 6,
    "Balloon": 5,
    "Spiky": 9,
    "Parabola": 9,
    "Biconvex": 5,
    "Cylinder": 5
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
    mesh_res_vals = [2]
    # smoothing_methods = ["jacobi", "khop", "invhop"]
    smoothing_methods = ["jacobi"]
    constrained_flags = [False, True]
    smoothing_ks = [0, 1, 3, 5]
    doublesided_flags = [1]  # New: enable both one-sided and double-sided
    trials = 1
    jobs = []

    for mesh_res in mesh_res_vals:
        for obj in reversed(objects):
            alpha = alpha_override[obj]
            ntrials = trials if mesh_res == 2 else 1
            for mode in projection_modes:
                for method in smoothing_methods:
                    for constrained in constrained_flags:
                        k_vals = smoothing_ks #if method in {"jacobi", "khop"} else [0]
                        for k in k_vals:
                            # Adjust learning rate and momentum based on k
                            if k == 1:
                                lr, momentum = 1e-5, 0.9
                            elif k == 3:
                                lr, momentum = 2e-5, 0.9
                            elif k == 5:
                                lr, momentum = 5e-5, 0.9
                            else:
                                lr, momentum = 1e-5, 0.9  # default

                            # Adjust number of iterations based on k
                            n_iters = 100

                            for doublesided in doublesided_flags:
                                for trial in range(ntrials):
                                    vis = int(trial == 0)
                                    run_name = f"{obj}_res{mesh_res}_{method}_k{k}_constr{int(constrained)}_dbl{doublesided}_t{trial:02d}"

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
                                        project=f"Parabola Smoothing",
                                        n_iters=n_iters,
                                        lr=lr,
                                        momentum=momentum,
                                        view_mode="manual",
                                        num_views=12,
                                        doublesided=doublesided,  # New: set flag
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
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()
    main(args.n_workers)
