from .dataloader import *
from .experimentrunner import *
import warnings
warnings.filterwarnings('ignore')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device("cpu") 
    data_path = "/home/jeffrey/honours/src/framework_mesh/skyconfig_gpu.json"
    with open(data_path, 'r') as f:
        config = json.load(f)

    # Modify the 'mesh_res' value
    config['paths']['mesh_res'] = 3
    dataloader = DataLoader(config, device)
    runner = ExperimentRunner("/home/jeffrey/honours/src/framework_mesh/exp_pipeline.json", dataloader)
    runner.run()

    # config['paths']['mesh_res'] = 3
    # dataloader = DataLoader(config)
    # runner = ExperimentRunner("/home/jeffrey/honours/src/experiments/exp_pipeline.json", dataloader)
    # runner.run()


if __name__ == "__main__":
    main()