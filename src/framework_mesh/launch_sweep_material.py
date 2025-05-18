import argparse
import subprocess
from multiprocessing import Pool

# --- CONFIG ---
data_path = "./framework_mesh/data_diffuse.json"
exp_path = "./framework_mesh/exp_all_diffuse.json"
device = "cpu"

shapes = ["Diamond", "Balloon", "Strawberry", "Spiky", "Parabola",  "Cube", "Cylinder"]
materials = ["Specular", "Real"]
mesh_res = 2
view_mode = "manual"
num_views = 12

alpha_override = {
    "Diamond": 2,
    "Balloon": 2,
    "Strawberry": 2,
    "Spiky": 10,
    "Parabola": 10,
    "Cube": 5,
    "Cylinder": 2,
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
            alpha = alpha_override[shape]
            for trial in range(1):  # One trial per config
                vis = 1  # Always enable vis
                cmd_base = dict(
                    data_path=get_data_path(material),
                    exp_base_path=exp_path,
                    target_object=shape,
                    projection_mode="alpha",
                    mesh_res=mesh_res,
                    alpha=alpha,
                    name=None,  # set later
                    device=device,
                    vis_enabled=vis,
                    velocity_enabled=0,
                    smoothing_enabled=0,
                    project=None,
                    view_mode=view_mode,
                    num_views=num_views,
                    n_iters=100,
                    lr=5e-6,
                    momentum=0.8,
                )

                # Run double-sided for all materials
                cmd = cmd_base.copy()
                cmd["doublesided"] = 1
                cmd["name"] = f"{shape}_{material}_dbl_t00"
                cmd["project"] = f"Material Sweep"
                cmd["material"] = material
                jobs.append(make_args(**cmd))

                # If Real, also run one-sided
                if material == "Real":
                    cmd = cmd_base.copy()
                    cmd["doublesided"] = 0
                    cmd["name"] = f"{shape}_{material}_one_t00"
                    cmd["project"] = f"Material Sweep"
                    cmd["material"] = material
                    jobs.append(make_args(**cmd))
    return jobs

def main(n_workers):
    jobs = sweep_materials()
    print(f"Launching {len(jobs)} jobs with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        pool.map(run_process, jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=3)
    args = parser.parse_args()
    main(args.n_workers)
