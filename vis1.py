import argparse
import os

import pandas as pd

pd.options.plotting.backend = "plotly"
import analysis.data_importing as imp  # Custom importing module
import analysis.plotting as pl  # Custom plotting module
import interfaces.postprocessing as pif  # post processing interface

parser = argparse.ArgumentParser(
                    prog='vis1',
                    description='',
                    epilog='')
parser.add_argument('activity_dir', type = str)


args = parser.parse_args()
#activity_data_dir = "../data/Juan_2023-03-18_Testing_Live_VE_and_Garmin"

activity_data_dir = args.activity_dir

uncleaned_data_dir = os.path.join(activity_data_dir, "uncleaned_data")
cleaned_data_dir = os.path.join(activity_data_dir, "cleaned_data")
imp.clean_all_data(uncleaned_data_dir)
clean_dfs = imp.load_cleaned_data(cleaned_data_dir)

raw_slow_df = clean_dfs["raw_slow_df"]
aws_b3_df = clean_dfs["aws_b3_df"]
live_b3_df = clean_dfs["live_b3_df"]

figs = []

raw_fig_dict = {
    "Raw Chest": (raw_slow_df, "time", "c")
}

VE_fig_dict = {
    "Post Processing": (aws_b3_df, "breathTime", "VE"),
    "Live": (live_b3_df, "breathTime", "VE"),
}
figs.append(
    pl.create_subplots_w_raw(VE_fig_dict, raw_fig_dict, plottitle= "VE (Top), Raw Chest (Bottom)", xtitle="Time [s]", ytitle1="VE", ytitle2="Raw Chest",
                             show=False))

VT_fig_dict = {
    "Post Processing": (aws_b3_df, "breathTime", "VT"),
    "Live": (live_b3_df, "breathTime", "VT"),
}
figs.append(
    pl.create_subplots_w_raw(VT_fig_dict, raw_fig_dict, plottitle= "VT (Top), Raw Chest (Bottom)", xtitle="Time [s]", ytitle1="VT", ytitle2="Raw Chest",
                             show=False))

RR_figs_dict = {
    "Post Processing": (aws_b3_df, "breathTime", "RRAvg"),
    "Live": (live_b3_df, "breathTime", "RRAvg")
}
figs.append(
    pl.create_subplots_w_raw(RR_figs_dict, raw_fig_dict, plottitle= "RRAvg (Top), Raw Chest (Bottom)", xtitle="Time [s]", ytitle1="VT", ytitle2="Raw Chest",
                             show=False))
pl.figures_to_html(figs, filename=os.path.join(activity_data_dir, "VE-VT.html"), show=True)
