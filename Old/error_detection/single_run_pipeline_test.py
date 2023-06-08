"""
This will run unit tests on the functions required for the single run pipeline.
"""
import os
import local_utils as lu


ACTIVITY = "0fc29afc-f68e-46dc-9959-45605fdd81b6"
DATA_DIR = "/Old/data\dataset"
ACTIVITY_DIR = os.path.join(DATA_DIR, ACTIVITY)

def get_live_from_json(json_path):
    data = lu.get_live_from_act_dir(json_path)
    return data

if __name__ == '__main__':
    data = get_live_from_json(ACTIVITY_DIR)
