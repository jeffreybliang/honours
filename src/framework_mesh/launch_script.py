# launcher.py
import subprocess
from multiprocessing import Pool

EXPERIMENTS = [
    "./framework_mesh/exp_sphere0.json",
    "./framework_mesh/exp_spherediff.json"
]

RESOLUTIONS = [2, 3]
CONSTRAINED = [False]
GROUND_OPTIONS = ["ground", "noground"]

def launch_job(args):
    exp_path, res, constrained, ground = args
    cmd = [
        "python", "-m", "framework_mesh.worker",
        "--exp_path", exp_path,
        "--mesh_res", str(res),
        "--ground", ground
    ]
    if constrained:
        cmd.append("--constrained")

    print(f"[LAUNCHING] {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    jobs = [
        (exp_path, res, constr, ground)
        for exp_path in EXPERIMENTS
        for res in RESOLUTIONS
        for constr in CONSTRAINED
        for ground in GROUND_OPTIONS
    ]

    with Pool(processes=2) as pool:
        pool.map(launch_job, jobs)
