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
#import analysis.plotting as pl  # Custom plotting module
import interfaces.postprocessing as pif  # post processing interface
from interfaces.postprocessing import BR_rVE_RTformat_wrapper_peak_detection as pp_peak_detection
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

def cross_corr_align(df1, df2, ycol1, ycol2, xcol1, xcol2):
    """
    This function takes in two dataframes and aligns them based on the cross-correlation
    between two columns. The function returns the lag value and the aligned dataframe.
    """
    from copy import deepcopy
    # First we need to join and interpolate the two series
    df1 = pd.DataFrame(df1[ycol1].set_axis(df1[xcol1]))
    df2 = pd.DataFrame(df2[ycol2].set_axis(df2[xcol2]))
    df = df1.join(df2, how="outer", rsuffix="_2", lsuffix="_1") # join the two dataframes, index is the union
    df_interp = df.interpolate(method="index").fillna(value = 0) #
    sec_df = pd.DataFrame(np.arange(max(df_interp.index)), index=np.arange(max(df_interp.index)))
    df_sec1 = df_interp.join(sec_df, how="outer")
    df_sec = df_sec1.interpolate(method="index").fillna(value =0).drop(columns=[0])
    # drop indices that are not integers
    df_sec = df_sec[df_sec.index % 1 == 0]

    # Now we can compute the cross correlation
    ycol_1 = ycol1 + "_1"
    ycol_2 = ycol2 + "_2"
    corr, lags = cross_correlate(df_sec[ycol_1], df_sec[ycol_2])
    opt_lag = lags[np.argmax(corr)]

    # Copy df sec
    df_sec_shift = deepcopy(df_sec)
    # Shift the second column by the optimal lag
    df_sec_shift[df_sec.columns[0]] = df_sec_shift[df_sec.columns[0]].shift(opt_lag)

    return df_sec_shift, opt_lag, df_sec


if False:
    parser = argparse.ArgumentParser(
                        prog='vis1',
                        description='',
                        epilog='')
    parser.add_argument('activity_dir', type = str)


    args = parser.parse_args()
    #activity_data_dir = "../data/Juan_2023-03-18_Testing_Live_VE_and_Garmin"

    activity_data_dir = args.activity_dir
else:
    activity_data_dir = "data/Arnar_Larusson_2023-04-12_easy_ride"

uncleaned_data_dir = os.path.join(activity_data_dir, "uncleaned_data")
cleaned_data_dir = os.path.join(activity_data_dir, "cleaned_data")
#imp.clean_all_data(uncleaned_data_dir)
clean_dfs = imp.load_cleaned_data(cleaned_data_dir)

raw_slow_df = clean_dfs["raw_slow_df"]
pp_df = clean_dfs["aws_b3_df"]
live_df = clean_dfs["live_b3_df"]
raw_fast_df = clean_dfs["raw_fast_df"]


color_dict = {
    "raw": "#1f77b4",
    "pp": "#d62728",
    "c": "#2ca02c",
    "err": "#ff7f0e"
}

# Get Raw Chest Data
raw_chest = pd.DataFrame(raw_slow_df["c"]).set_index(raw_slow_df["time"])

# Get Post processing peaks for raw chest data
pp_pd_df = pp_peak_detection(uncleaned_data_dir) # pp_peak_detection is a wrapper function that calls actual functions within pp algorithm

# Get peak and valley values from raw
pp_pd_df["PeakVal"] = raw_chest.loc[pp_pd_df["PeakTS"]]["c"].values
pp_pd_df["ValleyVal"] = raw_chest.loc[pp_pd_df["ValleyTS"]]["c"].values

# Align VT data
VT_sec_shift, opt_lag, VT_sec = cross_corr_align(live_df, pp_df, "VT", "VT", "breathTime", "breathTime")
VT_sec_shift.columns = ["VT live", "VT pp"]


## Traces
raw_chest_trace = go.Scatter(
    x = raw_chest.index,
    y = raw_chest["c"],
    mode = "lines",
    name = "Raw Chest",
    legendgroup= 1,
    line = {"color" : color_dict["raw"]}
)

peak_val_pp_trace = go.Scatter(
    x = pp_pd_df["PeakTS"],
    y = pp_pd_df["PeakVal"],
    mode = "markers",
    marker_symbol = "triangle-up",
    marker_size = 15,
    marker_color = color_dict["pp"],
    marker_line_width = 2,
    marker_line_color = "black",
    name = "Peak Val (PP)",
    legendgroup= 1,
    customdata= pp_pd_df.index,
    hovertemplate = "Peak Val (PP): %{y} <br> Peak Time: %{x} <br> Index: %{customdata}"
)
valley_val_pp_trace = go.Scatter(
    x = pp_pd_df["ValleyTS"],
    y = pp_pd_df["ValleyVal"],
    mode = "markers",
    marker_symbol = "triangle-down",
    marker_size = 15,
    marker_color = color_dict["pp"],
    marker_line_width = 2,
    marker_line_color = "black",
    legendgroup= 1,
    name = "Valley Val (PP)",
    customdata= pp_pd_df.index,
    hovertemplate = "Valley Val (PP): %{y} <br> Valley Time: %{x} <br> Index: %{customdata}"
)

VT_pp_trace = go.Scatter(
    x = pp_df["breathTime"],
    y = pp_df["VT"],
    mode = "lines",
    name = "VT (PP)",
    legendgroup= 2,
    line = {"color" : color_dict["pp"]}
)

VT_live_trace = go.Scatter(
    x = live_df["breathTime"],
    y = live_df["VT"],
    mode = "lines",
    name = "VT (Live)",
    legendgroup= 2,
    line = {"color" : color_dict["c"]}
)

VT_live_shift_trace = go.Scatter(
    x = VT_sec_shift.index,
    y = VT_sec_shift["VT live"],
    mode = "lines",
    name = "VT (Live) Shifted",
    legendgroup= 3,
    line = {"color" : color_dict["c"]}
)

VT_pp_shift_trace = go.Scatter(
    x = VT_sec_shift.index,
    y = VT_sec_shift["VT pp"],
    mode = "lines",
    name = "VT (PP) Shifted",
    legendgroup= 3,
    line = {"color" : color_dict["pp"]}
)


# Plot Traces
fig = make_subplots(rows = 3,cols=1, shared_xaxes=True, vertical_spacing=0.2,
                    subplot_titles=("Raw Chest", "VT")
                    )

## subplot 1: Raw chest with post processing peaks/valleys
fig.add_trace(raw_chest_trace, row = 1, col = 1)
fig.add_trace(peak_val_pp_trace, row = 1, col = 1)
fig.add_trace(valley_val_pp_trace, row = 1, col = 1)
fig.update_yaxes(title_text="Raw Chest", row=1, col=1)


## subplot 2: VT from post processing and live
fig.add_trace(VT_pp_trace, row = 2, col = 1)
fig.add_trace(VT_live_trace, row = 2, col = 1)
fig.update_yaxes(title_text="VT", row=2, col=1)

## subplot 3: VT from post processing and live shifted
fig.add_trace(VT_pp_shift_trace, row = 3, col = 1)
fig.add_trace(VT_live_shift_trace, row = 3, col = 1)
fig.update_yaxes(title_text="VT Live Shifted", row=3, col=1)


fig.update_layout(height=500*3,
                    legend_tracegroupgap=450,
                    )
fig.show(renderer = "browser")


if False:
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
    VT_sec_int["VT_l_shift"] = VT_sec_int["VT_l"].shift(-opt_lag)

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


    # Setup plot dicts

    plot_dicts = []
    xtitles = []
    ytitles = []

    # Get Raw Chest Data
    raw_chest = pd.DataFrame(raw_slow_df["c"]).set_index(raw_slow_df["time"])
    plot_dict_chest = {
        "Raw Chest" : (raw_chest, "index", "c")
    }
    plot_dicts.append(plot_dict_chest)
    xtitles.append("Time [s]")
    ytitles.append("Raw Chest")

    # Create VT plot dict
    plot_dict_VT = {
        "VT_pi": (VT_sec_int, "index", "VT_pi"),
        "VT_l": (VT_sec_int, "index", "VT_l_shift"),
        "Error" : (VT_sec_int, "index", "error")
    }
    plot_dicts.append(plot_dict_VT)
    xtitles.append("Time [s]")
    ytitles.append("VT / Error")

    # Get running average of the error
    VT_sec_int["running_avg_error"] = VT_sec_int["error"].rolling(window=60).mean()
    # Plot the running average of the error with the VT data
    plot_dict_ra = {
        "Minute Average Error" : (VT_sec_int, "index", "running_avg_error")}
    plot_dicts.append(plot_dict_ra)
    xtitles.append("Time [s]")
    ytitles.append("Minute Average Error")

    # Create VE plot dict
    plot_dict_VE = {
        "VE_pi": (VE_sec_int, "index", "VE_pi"),
        "VE_l_shift": (VE_sec_int, "index", "VE_l_shift"),
        "Error" : (VE_sec_int, "index", "error")
    }
    plot_dicts.append(plot_dict_VE)
    xtitles.append("Time [s]")
    ytitles.append("VE / Error")

    # Get Gyro
    gyr = raw_fast_df[["gx", "gy", "gz"]].set_index(raw_fast_df["time"])
    plot_dict_gyr = {
        "gx" : (gyr, "index", "gx"),
        "gy" : (gyr, "index", "gy"),
        "gz" : (gyr, "index", "gz")
    }
    plot_dicts.append(plot_dict_gyr)
    xtitles.append("Time [s]")
    ytitles.append("Gyro [deg/s]")

    # Get Accel
    acc = raw_fast_df[["ax", "ay", "az"]].set_index(raw_fast_df["time"])
    plot_dict_acc = {
        "ax" : (acc, "index", "ax"),
        "ay" : (acc, "index", "ay"),
        "az" : (acc, "index", "az")
    }
    plot_dicts.append(plot_dict_acc)
    xtitles.append("Time [s]")
    ytitles.append("Accel [g]")

    # Get pl data
    raw_pl = pd.DataFrame(raw_fast_df["pl"]).set_index(raw_fast_df["time"])
    plot_dict_pl = {
        "PL" : (raw_pl, "index", "pl")
    }
    plot_dicts.append(plot_dict_pl)
    xtitles.append("Time [s]")
    ytitles.append("pl")

    pl.create_subplots(plot_dicts, plottitle= "Vis2 Analysis", xtitles=xtitles, ytitles=ytitles, show= True)




