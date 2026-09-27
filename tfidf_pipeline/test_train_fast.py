import os
import sys
import pandas as pd
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import train_fast

# Override caps to run extremely fast
train_fast.S1_TRAIN_CAP = 5000
train_fast.S2_TRAIN_CAP = 5000
train_fast.S3_TRAIN_CAP = 5000

import config
config.NUM_WORKERS = 1  # Disable multiprocessing for test

def run():
    print("Testing train_fast.py end-to-end with tiny caps...")
    data_dir = os.path.join(_HERE, "..", "student_resource", "dataset")
    out_dir = os.path.join(_HERE, "test_output")
    
    # We want to use small subsets for the test set too.
    # We can mock the read_csv in train_fast to only read small chunks, but
    # modifying the train_fast script's pandas is easier.
    original_read_csv = pd.read_csv
    def mock_read_csv(filepath, **kwargs):
        kwargs['nrows'] = 5000
        return original_read_csv(filepath, **kwargs)
        
    pd.read_csv = mock_read_csv
    
    try:
        train_fast.run_pipeline(data_dir, out_dir)
        print("End-to-end test passed!")
    except Exception as e:
        print("End-to-end test FAILED!")
        import traceback
        traceback.print_exc()
    finally:
        pd.read_csv = original_read_csv

if __name__ == "__main__":
    run()
