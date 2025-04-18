from .dataloader import *
from .experimentrunner import *
import warnings
warnings.filterwarnings('ignore')

def main():
    dataloader = DataLoader("/home/jeffrey/honours/src/experiments/skyconfig.json")
    runner = ExperimentRunner("/home/jeffrey/honours/src/experiments/experiment_template.json", dataloader)
    runner.run()


if __name__ == "__main__":
    main()