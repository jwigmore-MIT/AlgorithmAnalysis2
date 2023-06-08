"""
This file will handle all the functions that will be unique to running the error analysis locally.
This will mostly be on the data retrieval and processing side.
"""

import os
import json
import pandas as pd
import numpy as np
from globals import METRIC_MAP as MM
from typing import Dict, List,


def get_25Hz_from_samples(samples: list(Dict) , chest_only: bool  = False) -> Dict:
    """
    This function will take in a list of samples and return a dictionary of the 25Hz data.
    :param samples: each element in the samples (list) is a dictionary of values
    :param chest_only: if true, only return the time and chest data
    :return: sample_dict: a dictionary of the 25Hz data
    """
    sample_names = ["t","id","p","alt","c","cr"]
    if chest_only:
        sample_names = ["t", "c"]
    sample_dict = {name: [] for name in sample_names}
    for sample in samples:
        for name in sample_names:
            sample_dict[name].append(sample[name])
    return sample_dict

def get_125Hz_from_samples(samples):
    sample_names = ['ax','ay','az','gx','gy','gz','mx','my','mz','pl',]
    sample_dict = {name: [] for name in sample_names}
    for sample in samples:
        for name in sample_names:
            sample_dict[name].extend(sample[name])
    return sample_dict

def get_raw_from_json(raw_json_path: str, get_slow: bool = True,
                      get_fast: bool = False, reorder=True, process=True) -> Dict:
    raw = json.load(open(raw_json_path, 'rb'))
    if not process:
        return raw
    # Processing samples (Raw Data)
    samples = raw['samples']
    if get_slow:
        slow_data = get_25Hz_from_samples(samples)
    if get_fast:
        fast_data = get_125Hz_from_samples(samples)

    # Get current time-stamps
    f_fast = 125  # Hz
    f_slow = 25  # Hz
    t_fast = [x / f_fast for x in range(len(fast_data['ax']))]
    t_slow = [x / f_slow for x in range(len(slow_data['id']))]
    fast_data['time'] = t_fast
    slow_data['time'] = t_slow

    fast_df = pd.DataFrame(fast_data)
    fast_cols = fast_df.columns.to_list()
    fast_cols = fast_cols[-1:] + fast_cols[:-1]
    fast_df = fast_df[fast_cols]

    slow_df = pd.DataFrame(slow_data)
    slow_cols = slow_df.columns.to_list()
    slow_cols = slow_cols[-1:] + slow_cols[:-1]
    slow_df = slow_df[slow_cols]

    ## Processing 'rtBreathing' (Live Algo Output)
    rtBreathing = raw["rtBreathing"]
    if len(rtBreathing) > 0:
        rtData = {}
        is_v37 = raw["info"]['fw_v'] == '0.37'
        if is_v37:
            Exception("Data is from Live Algorithm 0.37 -- Need to divide metrics by 100")
        for measurement, data in rtBreathing[0].items():
            rtData[measurement] = []
        for data_dict in rtBreathing:
            for measurement, data in data_dict.items():
                rtData[measurement].append(int(data))
        rt_df = pd.DataFrame(rtData)
        if reorder:

            order = [name for name in MM.keys() if name in rt_df.columns]
            order.extend([name for name in rt_df.columns if name not in order])
            rt_df = rt_df[order]
    else:
        print("")
        rt_df = None

    return fast_df, slow_df, rt_df
def get_live_from_act_dir(act_dir, get_slow = True, get_fast = False):
    """
    This will get all the live data from the activity directory.
    :param act_dir: The activity directory
    :return: live data
    """
    json_file_paths = []
    for file_name in os.listdir(act_dir):
        if file_name.endswith('.json'):
            json_file_paths.append(os.path.join(act_dir, file_name))
    if len(json_file_paths) > 1:
        print("RECHECK ACTIVITY DIRECTORY - MULTIPLE JSONS FOUND")
    else:
        raw_json_path = json_file_paths[0]
    data = get_raw_from_json(raw_json_path, get_slow = get_slow, get_fast = get_fast,  reorder=True, process=True)
    return data["live"]