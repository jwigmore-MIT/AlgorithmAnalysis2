from main_functions import method1_error_analysis, plot_all_signals
from data_retrieval import load_data, load_data2
import os
import numpy as np

# -------------------------------------#
# Parameters
# -------------------------------------#
# Dataset specification
dataset_dir = "dataset/" # location of all datasets
dataset = "2b8f256d-7063-410c-b66d-3bdfb7d140c5" # name of dataset folder within directory
activity_dir = dataset_dir + dataset # location of activity folder

# whether or not to use live c data (csv)
live_csv = True
live_json = not live_csv
pp_csv = True
live_df_name = "live_b3_df" if live_json else "c_df"

key_metrics = ["breathTime", "VT", "instBR", "VE"] # metrics to extract from dataset (always include breathTime)
plot = True # whether or not to plot signals and their errors

# Time parameters
min_time = 120 # the minimum time (s) to
max_time = np.infty # if negative (-t), will set max_time = max_index - t
live_shift = -6 # How much to shift the live signal by (should be negative)

# Error parameters
error_window = 3 # search window for error signal
error_threshold = 0.5 # threshold for error signal
RAT_error_window = 20 # search window for RAT error signal
RAT_error_threshold = 0.25 # threshold for RAT error signal
# -------------------------------------#
# End Parameters
# -------------------------------------#



# Data loading
activity = os.path.split(activity_dir)[-1]
clean_dfs = load_data2(activity_dir, live_json, live_csv, pp_csv)
raw_chest = clean_dfs["raw_slow_df"][["time","c",]]
if max_time < 0:
    max_time = raw_chest["time"].max() + max_time

raw_chest = raw_chest[(raw_chest["time"] > min_time) & (raw_chest["time"] < max_time)]
live_b3_df = clean_dfs[live_df_name][key_metrics]
pp_b3_df = clean_dfs["aws_b3_df"][key_metrics]

results = {}
for metric in key_metrics:
    if metric == "breathTime":
        continue
    results["metric"] = method1_error_analysis(live_b3_df, pp_b3_df, metric,
                                    return_all = True,
                                    live_shift=live_shift,
                                    min_time = min_time,
                                    max_time = max_time,
                                    error_window = error_window,
                                    error_threshold = error_threshold,
                                    RAT_error_window = RAT_error_window,
                                    RAT_error_threshold = RAT_error_threshold)

    if plot:
        plot_all_signals(raw_chest, results["metric"], metric, dataset = dataset)