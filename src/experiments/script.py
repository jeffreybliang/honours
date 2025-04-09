from src.dataloader import *
from src.experimentrunner import *
import warnings
warnings.filterwarnings('ignore')

def main():
    # from source S to target T, vary the number of views randomly
    dataloader = DataLoader("/home/jeffrey/honours/src/experiments/skyconfig_balloon.json")
    runner = ExperimentRunner("/home/jeffrey/honours/src/experiments/experiment_spline.json", dataloader)
    runner.run()

if __name__ == "__main__":
    main()