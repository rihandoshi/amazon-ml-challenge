import os
import sys
import pandas as pd
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import train_ensemble

# Set caps for the 50k test run
train_ensemble.S1_TRAIN_CAP = 50000
train_ensemble.S2_TRAIN_CAP = 150000
train_ensemble.S3_TRAIN_CAP = 150000

# We don't disable multiprocessing this time to get a real timing estimate
# config.NUM_WORKERS is untouched (defaults to 16 or whatever is in config)

def run():
    print("Testing train_ensemble.py with 50k caps to check recall and timing...")
    data_dir = os.path.join(_HERE, "..", "student_resource", "dataset")
    out_dir = os.path.join(_HERE, "test_output_50k")
    
    # Run pipeline up to end of STAGE 2 by monkey-patching or just letting it run and we can kill it
    train_ensemble.run_pipeline(data_dir, out_dir)

if __name__ == "__main__":
    run()
