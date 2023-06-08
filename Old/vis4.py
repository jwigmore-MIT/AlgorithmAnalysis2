import argparse
import os
import pandas as pd
import numpy as np
pd.options.plotting.backend = "plotly"
import plotly.io as pio
pio.renderers.default = "jupyterlab"
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import analysis.data_importing as imp  # Custom importing module
import analysis.plotting as pl  # Custom plotting module
import interfaces.postprocessing as pif  # post processing interface
import scipy.signal
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def cross_correlate(series1, series2):
    sig1 = series1.dropna()
    sig2 = series2.dropna()
    corr = scipy.signal.correlate(sig1, sig2)
    lags = scipy.signal.correlation_lags(len(sig1), len(sig2))

    return corr / corr.max(), lags


def plot_cross_corr(series1, series2, corr, lags, title="", show=False, renderer = ""):
    fig = make_subplots(rows=2, cols=1)
    fig.update_layout(title=title)
    fig.update_xaxes(title = "Time [s]", row=1, col=1)
    fig.update_yaxes(title = "VT", row=1, col=1)
    fig.update_xaxes(title = "Lag [s]", row=2, col=1)
    fig.update_yaxes(title = "Cross Correlation", row=2, col=1)
    fig = fig.add_trace(go.Scatter(
        x=series1.index,
        y=series1.values,
        name="Series1"
    ), row=1, col=1)
    fig = fig.add_trace(go.Scatter(
        x=series2.index,
        y=series2.values,
        name="Series2"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=lags,
        y=corr,
        name="Cross-Correlation"
    ), row=2, col=1)
    if show:
        fig.show()
    return fig

# parser = argparse.ArgumentParser(
#                     prog='vis1',
#                     description='',
#                     epilog='')
# parser.add_argument('activity_dir', type = str)
#
#
# args = parser.parse_args()
activity_data_dir = "data/effad897-2991-4c2c-9a6a-f85a111d0e3d"

#activity_data_dir = args.activity_dir

uncleaned_data_dir = os.path.join(activity_data_dir, "uncleaned_data")
cleaned_data_dir = os.path.join(activity_data_dir, "cleaned_data")
imp.clean_all_data(uncleaned_data_dir)
clean_dfs = imp.load_cleaned_data(cleaned_data_dir)

raw_slow_df = clean_dfs["raw_slow_df"]
aws_b3_df = clean_dfs["aws_b3_df"]
live_b3_df = clean_dfs["live_b3_df"]
raw_fast_df = clean_dfs["raw_fast_df"]

# Get and Clean Data
VT_p = pd.DataFrame(aws_b3_df["VT"].set_axis(aws_b3_df["breathTime"]))
VT_pi = pd.DataFrame(aws_b3_df["VT"].set_axis(aws_b3_df["breathTime"].astype(int)))
VT_l = pd.DataFrame(live_b3_df["VT"].set_axis(live_b3_df["breathTime"].astype(int)))

# Interpolate VT Data and resample to second by second
VT_j = VT_pi.join(VT_l, how="outer", lsuffix="_pi", rsuffix="_l")
VT_j_int = VT_j.interpolate(method="index").fillna(value = 0) #fillna needed due to mismatch in length of each VT series
d = pd.DataFrame(np.arange(max(VT_j_int.index)), index = np.arange(max(VT_j_int.index)))
VT_sec = VT_j_int.join(d, how="outer")
VT_sec_int = VT_sec.interpolate(method="index").fillna(value = 0).drop(columns=[0])

# Compute cross correlation to find optimal lag
corr, lags = cross_correlate(VT_sec_int["VT_pi"], VT_sec_int["VT_l"])
opt_lag = lags[np.argmax(corr)]
VT_sec_int["VT_l_shift"] = VT_sec_int["VT_l"].shift(opt_lag)

# Get error between post processing and shifted live data
VT_sec_int["error"] = VT_sec_int["VT_pi"] - VT_sec_int["VT_l_shift"]

# Now do the same with the minute volume data
VE_p = pd.DataFrame(aws_b3_df["VE"]).set_axis(aws_b3_df["breathTime"])
VE_pi = pd.DataFrame(aws_b3_df["VE"]).set_axis(aws_b3_df["breathTime"].astype(int))
VE_l = pd.DataFrame(live_b3_df["VE"]).set_axis(live_b3_df["breathTime"])
VE_j = VE_pi.join(VE_l, how="outer", rsuffix="_l", lsuffix="_pi")
VE_j_int = VE_j.interpolate(method="index")
d = pd.DataFrame(np.arange(max(VE_j_int.index)), index = np.arange(max(VE_j_int.index)))
VE_sec = VE_j_int.join(d, how="outer")
VE_sec_int = VE_sec.interpolate(method="index").fillna(value = 0).drop(columns=[0])
VE_sec_int["VE_l_shift"] = VE_sec_int["VE_l"].shift(opt_lag)
VE_sec_int["error"] = VE_sec_int["VE_pi"] - VE_sec_int["VE_l_shift"]

## Get Breathing Rate Data
RR_p = pd.DataFrame(aws_b3_df["instBR"].set_axis(aws_b3_df["breathTime"]))
RR_pi = pd.DataFrame(aws_b3_df["instBR"].set_axis(aws_b3_df["breathTime"].astype(int)))
RR_l = pd.DataFrame(live_b3_df["instBR"].set_axis(live_b3_df["breathTime"].astype(int)))

# Interpolate RR data and resample to second by second
RR_j = RR_pi.join(RR_l, how="outer", lsuffix="_pi", rsuffix="_l")
RR_j_int = RR_j.interpolate(method="index").fillna(value = 0) #fillna needed due to mismatch in length of each VT series
d = pd.DataFrame(np.arange(max(RR_j_int.index)), index = np.arange(max(RR_j_int.index)))
RR_sec = RR_j_int.join(d, how="outer")
RR_sec_int = RR_sec.interpolate(method="index").fillna(value = 0).drop(columns=[0])

# Compute cross correlation to find optimal lag
corr, lags = cross_correlate(RR_sec_int["instBR_pi"], RR_sec_int["instBR_l"])
opt_lag = lags[np.argmax(corr)]
RR_sec_int["instBR_l_shift"] = RR_sec_int["instBR_l"].shift(opt_lag)

# Get error between post processing and shifted live data
RR_sec_int["error"] = RR_sec_int["instBR_pi"] - RR_sec_int["instBR_l_shift"]





# Setup plot dicts

plot_dicts = []
xtitles = []
ytitles = []
subplot_titles = []

# Get Raw Chest Data
raw_chest = pd.DataFrame(raw_slow_df["c"]).set_index(raw_slow_df["time"])
plot_dict_chest = {
    "Raw Chest" : (raw_chest, "index", "c")
}
plot_dicts.append(plot_dict_chest)
xtitles.append("Time [s]")
ytitles.append("Raw Chest")
subplot_titles.append("Raw Chest")

# Create VT plot dict
plot_dict_VT = {
    "VT PP": (VT_sec_int, "index", "VT_pi"),
    "VT Live": (VT_sec_int, "index", "VT_l_shift"),
    "Error" : (VT_sec_int, "index", "error")
}
plot_dicts.append(plot_dict_VT)
xtitles.append("Time [s]")
ytitles.append("VT / Error")
subplot_titles.append("VT and VT Error")

# Create RR plot dict
plot_dict_RR = {
    "RR PP": (RR_sec_int, "index", "instBR_pi"),
    "RR Live": (RR_sec_int, "index", "instBR_l_shift"),
    "Error" : (RR_sec_int, "index", "error")
}
plot_dicts.append(plot_dict_RR)
xtitles.append("Time [s]")
ytitles.append("RR / Error")
subplot_titles.append("Instantaneous Breathing Rate Error")





# # Get running average of the error
# VT_sec_int["running_avg_error"] = VT_sec_int["error"].rolling(window=60).mean()
# # Plot the running average of the error with the VT data
# plot_dict_ra = {
#     "Minute Average Error" : (VT_sec_int, "index", "running_avg_error")}
# plot_dicts.append(plot_dict_ra)
# xtitles.append("Time [s]")
# ytitles.append("Minute Average Error")
# subplot_titles.append("Running 60s Average of the VT Error Signal")

# Create VE plot dict
plot_dict_VE = {
    "VE PP": (VE_sec_int, "index", "VE_pi"),
    "VE Live": (VE_sec_int, "index", "VE_l_shift"),
    "Error" : (VE_sec_int, "index", "error")
}
plot_dicts.append(plot_dict_VE)
xtitles.append("Time [s]")
ytitles.append("VE / Error")
subplot_titles.append("VE and VE Error")

# Get Gyro
# gyr = raw_fast_df[["gx", "gy", "gz"]].set_index(raw_fast_df["time"])
# plot_dict_gyr = {
#     "gx" : (gyr, "index", "gx"),
#     "gy" : (gyr, "index", "gy"),
#     "gz" : (gyr, "index", "gz")
# }
# plot_dicts.append(plot_dict_gyr)
# xtitles.append("Time [s]")
# ytitles.append("Gyro [deg/s]")
#
# # Get Accel
# acc = raw_fast_df[["ax", "ay", "az"]].set_index(raw_fast_df["time"])
# plot_dict_acc = {
#     "ax" : (acc, "index", "ax"),
#     "ay" : (acc, "index", "ay"),
#     "az" : (acc, "index", "az")
# }
# plot_dicts.append(plot_dict_acc)
# xtitles.append("Time [s]")
# ytitles.append("Accel [g]")

# Get pl data
raw_pl = pd.DataFrame(raw_fast_df["pl"]).set_index(raw_fast_df["time"])
plot_dict_pl = {
    "PL" : (raw_pl, "index", "pl")
}
plot_dicts.append(plot_dict_pl)
xtitles.append("Time [s]")
ytitles.append("pl")
subplot_titles.append("Movement according to PL")

pl.create_subplots(plot_dicts, plottitle= "Vis2 Analysis", xtitles=xtitles, ytitles=ytitles, subplot_titles = subplot_titles, show= True)




