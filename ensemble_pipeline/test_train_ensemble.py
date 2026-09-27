import os
import sys
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import train_ensemble

# Override caps to run extremely fast for test
train_ensemble.S1_TRAIN_CAP = 2000
train_ensemble.S2_TRAIN_CAP = 2000
train_ensemble.S3_TRAIN_CAP = 2000

import config
config.NUM_WORKERS = 1  # Disable multiprocessing for test

def run():
    print("Testing train_ensemble.py end-to-end with tiny caps...")
    data_dir = os.path.join(_HERE, "..", "student_resource", "dataset")
    out_dir = os.path.join(_HERE, "test_output")
    
    # Mock read_csv to only read small chunks
    original_read_csv = pd.read_csv
    def mock_read_csv(filepath, **kwargs):
        kwargs['nrows'] = 2000
        return original_read_csv(filepath, **kwargs)
        
    pd.read_csv = mock_read_csv
    
    try:
        train_ensemble.run_pipeline(data_dir, out_dir)
        print("End-to-end test passed!")
    except Exception as e:
        print("End-to-end test FAILED!")
        import traceback
        traceback.print_exc()
    finally:
        pd.read_csv = original_read_csv

if __name__ == "__main__":
    run()
