from src.dataloader import *
from src.experimentrunner import *
import warnings
warnings.filterwarnings('ignore')

def main():
    # from source S to target T, vary the number of views randomly
    dataloader = DataLoader("/home/jeffrey/honours/src/experiments/skyconfig_edge1.json")

    with open("/home/jeffrey/honours/src/experiments/experiment_edges.json") as f:
        base_cfg = json.load(f)

    tgt_mesh_name = base_cfg["target_meshes"][0]
    total_views = len(base_cfg["views"][tgt_mesh_name]["view_idx"])

    for nviews in range(1, total_views + 1):
        cfg = base_cfg.copy()
        cfg["views"][tgt_mesh_name]["num_views"] = nviews
        base_name = f"{cfg['name']}{nviews}"
        print(f"\n[INFO] Starting experiments with {nviews} view(s)...")

        for r in range(8):
            cfg["name"] = f"{base_name}_r{r}"
            print(f"[INFO] Running: {cfg['name']}")
            runner = ExperimentRunner(cfg, dataloader)
            runner.run()
            print(f"[INFO] Finished: {cfg['name']}")

if __name__ == "__main__":
    main()