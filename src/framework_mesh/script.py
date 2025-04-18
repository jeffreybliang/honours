from .dataloader import *
from .experimentrunner import *
import warnings
warnings.filterwarnings('ignore')

def main():
    # from source S to target T, vary the number of views randomly
    dataloader = DataLoader("/home/jeffrey/honours/src/experiments/skyconfig_balloon.json")
    runner = ExperimentRunner("/home/jeffrey/honours/src/experiments/experiment_spline.json", dataloader)
    runner.run()
from .dataloader import *
from .experimentrunner import *
import warnings
warnings.filterwarnings('ignore')

def main():
    data_path = "/home/jeffrey/honours/src/experiments/skyconfig_local.json"
    with open(data_path, 'r') as f:
        config = json.load(f)

    # Modify the 'mesh_res' value
    config['paths']['mesh_res'] = 2
    dataloader = DataLoader(config)
    runner = ExperimentRunner("/home/jeffrey/honours/src/experiments/exp_pipeline.json", dataloader)
    runner.run()

    config['paths']['mesh_res'] = 3
    dataloader = DataLoader(config)
    runner = ExperimentRunner("/home/jeffrey/honours/src/experiments/exp_pipeline.json", dataloader)
    runner.run()


if __name__ == "__main__":
    main()