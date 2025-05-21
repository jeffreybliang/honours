import argparse
import subprocess
from multiprocessing import Pool

# --- CONFIG ---
data_path = "./framework_mesh/data_diffuse.json"
exp_path = "./framework_mesh/exp_all_diffuse.json"
device = "cpu"

shapes = ["Diamond", "Balloon", "Strawberry", "Spiky", "Parabola",  "Cube", "Cylinder", "Bottle", "Biconvex", "Uneven", "Tear", "Turnip",  "Ellipsoid", "Sponge"]
materials = ["Diffuse", "Specular", "Real"]
mesh_res = 2
view_mode = "manual"
num_views = 12

alpha_override = {
    "Diamond": 2,
    "Balloon": 2,
    "Strawberry": 2,
    "Spiky": 8,
    "Parabola": 8,
    "Cube": 5,
    "Cylinder": 2,
    "Bottle": 5,
    "Biconvex": 5,
    "Uneven": 10,
    "Tear": 10,
    "Turnip": 5,
    "Ellipsoid": 5,
    "Sponge": 10
}

def get_data_path(material):
    return f"./framework_mesh/data_{material.lower()}.json"


def make_args(**kwargs):
    args = ["python", "-m", "framework_mesh.worker"]
    for k, v in kwargs.items():
        args += [f"--{k}", str(v)]
    return args

def run_process(cmd):
    subprocess.run(cmd)

def sweep_materials():
    jobs = []
    for material in materials:
        for shape in shapes:
            alpha = alpha_override.get(shape, 5)
            for trial in range(1):  # Two trials
                vis = int(trial < 1)

                for doublesided in (1, 0):  # Sweep both modes
                    cmd = dict(
                        data_path=get_data_path(material),
                        exp_base_path=exp_path,
                        target_object=shape,
                        projection_mode="alpha",
                        mesh_res=mesh_res,
                        alpha=alpha,
                        name=f"{shape}_{material}_{'dbl' if doublesided else 'sngl'}_t{trial:02d}",
                        device=device,
                        vis_enabled=vis,
                        velocity_enabled=0,
                        smoothing_enabled=0,
                        project="REDO Material Sweep",
                        view_mode=view_mode,
                        num_views=num_views,
                        n_iters=100,
                        lr=1e-5 if doublesided else 5e-6,  # Adjust learning rate
                        momentum=0.9,
                        doublesided=doublesided,
                        material=material
                    )
                    jobs.append(make_args(**cmd))
    return jobs

def main(n_workers):
    jobs = sweep_materials()
    print(f"Launching {len(jobs)} jobs with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        pool.map(run_process, jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()
    main(args.n_workers)
