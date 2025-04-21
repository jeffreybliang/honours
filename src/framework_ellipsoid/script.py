import json
import argparse
from copy import deepcopy
from .experiments import EllipsoidExperiment
import warnings
warnings.filterwarnings('ignore')

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True, help="Path to base config")
    # parser.add_argument("--param", type=str, required=True, help="Parameter to vary, e.g. target.m")
    # parser.add_argument("--values", type=str, required=True, help="Comma-separated list of values")
    # args = parser.parse_args()

    # base_cfg = load_config(args.config)
    # values = [eval(v) for v in args.values.split(",")]

    # for v in values:
    #     cfg = deepcopy(base_cfg)

    #     # Support nested keys like "target.m"
    #     keys = args.param.split(".")
    #     sub_cfg = cfg
    #     for k in keys[:-1]:
    #         sub_cfg = sub_cfg[k]
    #     sub_cfg[keys[-1]] = v
    #     param_value_str = str(v).replace(" ", "").replace("[", "").replace("]", "").replace(",", "-")
    #     cfg["name"] = f"{cfg['problem']}-{args.param.replace('.', '_')}-{param_value_str}"

    #     print(f"Running with {args.param} = {v}")
        config_path = "/Users/jeffreyliang/Documents/Honours/honours/src/framework_ellipsoid/config_nonaxisaligned.json"
    
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        experiment = EllipsoidExperiment(cfg)
        experiment.run()

if __name__ == "__main__":
    main()
