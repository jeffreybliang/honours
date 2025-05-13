import subprocess
import sys
import torch

# Fixed experiment settings
device = "cpu"
exp_base_path = "./framework_mesh/exp_oblique_AdamW.json"
velocity_k = 1
velocity_beta = 1.0
fixed_config = {
    "lr": 5e-3,
    "weight_decay": 0.0,
    "beta1": 0.9,
    "beta2": 0.9
}

# Mesh & config variants
configs = [
    {
        "data_path": "./framework_mesh/data_noground.json",
        "mesh_res": 2,
        "doublesided": True,
        "ground_label": "noground",
        "constrained_vals": [False]
    },
    {
        "data_path": "./framework_mesh/data_noground.json",
        "mesh_res": 3,
        "doublesided": True,
        "ground_label": "noground",
        "constrained_vals": [False]
    },
    {
        "data_path": "./framework_mesh/data_ground.json",
        "mesh_res": 2,
        "doublesided": False,
        "ground_label": "ground",
        "constrained_vals": [False, True]
    },
    {
        "data_path": "./framework_mesh/data_ground.json",
        "mesh_res": 3,
        "doublesided": False,
        "ground_label": "ground",
        "constrained_vals": [False]
    }
]

# Concurrency
max_procs = 2
running_procs = []

def build_cmd(idx, sweep_cfg, constrained):
    name = (
        f"{sweep_cfg['ground_label']}_AdamW_idx{idx}"
        f"_lr{fixed_config['lr']}_wd{fixed_config['weight_decay']}"
        f"_b1{fixed_config['beta1']}_b2{fixed_config['beta2']}"
        f"_constrained{constrained}_res{sweep_cfg['mesh_res']}"
    )
    return [
        sys.executable, "-m", "framework_mesh.worker",
        "--data_path", sweep_cfg["data_path"],
        "--exp_base_path", exp_base_path,
        "--mesh_res", str(sweep_cfg["mesh_res"]),
        "--constrained", str(constrained).lower(),
        "--optimiser", "AdamW",
        "--lr", str(fixed_config["lr"]),
        "--weight_decay", str(fixed_config["weight_decay"]),
        "--beta1", str(fixed_config["beta1"]),
        "--beta2", str(fixed_config["beta2"]),
        "--velocity_k", str(velocity_k),
        "--velocity_beta", str(velocity_beta),
        "--doublesided", str(sweep_cfg["doublesided"]).lower(),
        "--ground_label", sweep_cfg["ground_label"],
        "--device", device,
        "--name", name
    ]

# Launch
idx = 0
for sweep_cfg in configs:
    for constrained in sweep_cfg["constrained_vals"]:
        cmd = build_cmd(idx, sweep_cfg, constrained)
        print(f"[LAUNCH] {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        running_procs.append(proc)
        idx += 1

        if len(running_procs) >= max_procs:
            for p in running_procs:
                p.wait()
            running_procs = []

# Final wait
for p in running_procs:
    p.wait()
