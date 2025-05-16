import subprocess
import sys
import torch
import time

# Fixed experiment settings
device = "cpu"
data_path = "./framework_mesh/data_noground.json"
exp_base_path = "./framework_mesh/exp_oblique_AdamW_TEST.json"
mesh_res = 2
velocity_k = 1
velocity_beta = 1.0
doublesided = False
constrained = True
ground_label = "ground"

# Grid sweep parameters
lrs = [1e-3, 5e-3, 1e-2]
weight_decays = [0.0, 1e-4]
beta1_vals = [0.9, 0.95]
beta2_vals = [0.9, 0.95]

# Generate all combinations
sweeps = []
for lr in lrs:
    for wd in weight_decays:
        for b1 in beta1_vals:
            for b2 in beta2_vals:
                sweeps.append({
                    "lr": lr,
                })

# Limit concurrent subprocesses
max_procs = 4
running_procs = []

def build_cmd(idx, sweep_cfg, constrained, projection_mode, alpha, object_name):
    name = (
        f"{object_name}_idx{idx}_proj{projection_mode}_alpha{alpha}_"
        f"lr{fixed_config['lr']}"
        f"constrained{constrained}_res{sweep_cfg['mesh_res']}"
    )
    return [
        sys.executable, "-m", "framework_mesh.worker",
        "--data_path", sweep_cfg["data_path"],
        "--exp_base_path", exp_base_path,
        "--mesh_res", str(sweep_cfg["mesh_res"]),
        "--constrained", str(constrained).lower(),
        "--optimiser", "SGD",
        "--lr", str(fixed_config["lr"]),
        "--velocity_k", str(velocity_k),
        "--velocity_beta", str(velocity_beta),
        "--doublesided", str(sweep_cfg["doublesided"]).lower(),
        "--ground_label", sweep_cfg["ground_label"],
        "--device", device,
        "--name", name,
        "--projection_mode", projection_mode,
        "--alpha", str(alpha),
        "--object_name", object_name
    ]

# Launch processes with concurrency limit
for idx, config in enumerate(sweeps):
    cmd = build_cmd(config, idx)
    print(f"[LAUNCH] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    running_procs.append(proc)

    if len(running_procs) >= max_procs:
        for p in running_procs:
            p.wait()
        running_procs = []

# Final cleanup
for p in running_procs:
    p.wait()
