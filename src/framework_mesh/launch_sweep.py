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
                    "weight_decay": wd,
                    "beta1": b1,
                    "beta2": b2
                })

# Limit concurrent subprocesses
max_procs = 4
running_procs = []

def build_cmd(config, idx):
    name = f"{ground_label}_AdamW_idx{idx}_lr{config['lr']}_wd{config['weight_decay']}_b1{config['beta1']}_b2{config['beta2']}_res{mesh_res}"
    return [
        sys.executable, "-m", "framework_mesh.worker",
        "--data_path", data_path,
        "--exp_base_path", exp_base_path,
        "--mesh_res", str(mesh_res),
        "--constrained", str(constrained).lower(),
        "--optimiser", "AdamW",
        "--lr", str(config["lr"]),
        "--weight_decay", str(config["weight_decay"]),
        "--beta1", str(config["beta1"]),
        "--beta2", str(config["beta2"]),
        "--velocity_k", str(velocity_k),
        "--velocity_beta", str(velocity_beta),
        "--doublesided", str(doublesided).lower(),
        "--ground_label", ground_label,
        "--device", device,
        "--name", name
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
