from config import config
from utils import IOHelper
from benchmark import benchmark

"""
Main entry of the program
loads the data and starts the benchmark.
All configurations (parameters) of this benchmark are specified in config.py
"""

def main():
    
    # load data
    trainX, trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)
    
    # start training
    benchmark(trainX, trainY)

if __name__=='__main__':
    main()