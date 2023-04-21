import os

import pandas as pd
import numpy as np
pd.options.plotting.backend = "plotly"
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


def plot_cross_corr(series1, series2, corr, lags, title="", show=False):
    fig = make_subplots(rows=2, cols=1)
    fig.update_layout(title=title)
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ## Setup
    # activity_data_dir = "User_data/Arnar_Larusson_2023-04-12_easy_ride"
    # activity_data_dir = "User_data/Juan_2023-03-18_Testing_Live_VE_and_Garmin"
    activity_data_dir = "../data/Juan_2023-03-18_Testing_Live_VE_and_Garmin"
    uncleaned_data_dir = os.path.join(activity_data_dir, "uncleaned_data")
    cleaned_data_dir = os.path.join(activity_data_dir, "cleaned_data")
    imp.clean_all_data(uncleaned_data_dir)
    clean_dfs = imp.load_cleaned_data(cleaned_data_dir)

    raw_slow_df = clean_dfs["raw_slow_df"]
    aws_b3_df = clean_dfs["aws_b3_df"]
    live_b3_df = clean_dfs["live_b3_df"]

    # Post Processing Data from BR_rVE_RTformat
    chest_raw, chest_5hz, chest_bs, chest_bs_smooth, time, X_bbyb_df = pif.BR_rVE_RTformat_wrapper(uncleaned_data_dir)


    ## Original VT Signals
    VT_p = pd.DataFrame(aws_b3_df["VT"].set_axis(aws_b3_df["breathTime"]))
    VT_pi = pd.DataFrame(aws_b3_df["VT"].set_axis(aws_b3_df["breathTime"].astype(int)))
    VT_l = live_b3_df["VT"].set_axis(live_b3_df["breathTime"].astype(int))
    #corr1, lags1 = cross_correlate(VT_S1, VT_S2)
    #fig1 = plot_cross_corr(VT_S1, VT_S2, corr1, lags1, title = "Original Signals", show = True)


    #
    # corr2, lags2 = cross_correlate(VT_S1i, VT_S2)
    # fig = plot_cross_corr(VT_S1i, VT_S2, corr2, lags2, title="Int S1", show=True)
    if False:
        VT_df1 = pd.DataFrame(VT_S1i, columns = ["VT_p"])
        VT_df2 = pd.DataFrame(VT_S2, columns = ["VT_l"])
        VT_j = VT_df1.join(VT_df2, how = "outer", lsuffix="1", rsuffix= "2") #pd.merge(VT_S1i, VT_S2, how = "outer")

        VT_j2 = VT_j.interpolate(method = "index").fillna(value = 0)
        corr, lags = cross_correlate(VT_j2["VT1"], VT_j2["VT2"])
        fig = plot_cross_corr(VT_j2["VT1"], VT_j2["VT2"], corr, lags, title="Interpolated", show=True)

        time = pd.RangeIndex(start = 0, stop = max(VT_j2.index))
        d = pd.DataFrame(np.arange(max(VT_j2.index)), index = np.arange(max(VT_j2.index)))
        VT_j3 = VT_j2.join(d, how = "outer", lsuffix = "l", rsuffix = "r").interpolate(method = "index").fillna(value = 0)
        corr, lags = cross_correlate(VT_j3["VT1"], VT_j3["VT2"])
        fig = plot_cross_corr(VT_j3["VT1"], VT_j3["VT2"], corr, lags, title="Second by Second Interpolated", show=True)

        opt_lag = lags[np.argmax(corr)]
        VT_j4 = VT_j3
        VT_j4["VT2"] = VT_j4["VT2"].shift(opt_lag)

        corr, lags = cross_correlate(VT_j4["VT1"], VT_j4["VT2"])
        fig = plot_cross_corr(VT_j4["VT1"], VT_j4["VT2"], corr, lags, title="Post Time shift", show=True)


    # corr = scipy.signal.correlate(aws_b3_df["VT"].dropna(), live_b3_df["VT"].dropna())
    # lags = scipy.signal.correlation_lags(len(aws_b3_df["VT"].dropna()), len(live_b3_df["VT"].dropna()))
    # corr_df = pd.DataFrame(np.array([lags, corr/corr.max()]).T, columns =["lags", "cross-correlation"])
    # fig = corr_df.plot(x = "lags", y = "cross-correlation")
    # fig.show()

    ## Compare
    if False:
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

        #figs.append(raw_slow_df.plot(x="time", y="c", title="Chest Raw vs Time"))
        #figs.append(pl.create_spectrogram(chest_raw, 25, "Chest Raw - Spectrogram"))

        #figs.append(pd.DataFrame(chest_5hz, columns=["Chest Bandpass"]).plot(title="Chest Bandpass (0.1-5Hz) vs Time"))
        #figs.append(pl.create_spectrogram(chest_5hz, 25, "Chest Bandpass (0.1-5Hz) - Spectrogram"))

        #figs.append(pd.DataFrame(chest_bs, columns=["Chest Bandstop (> 5Hz)"]).plot(title="Chest Bandstop (> 5Hz) vs Time"))
        #figs.append(pl.create_spectrogram(chest_bs, 25, "Chest Bandstop (> 5Hz) - Spectrogram"))

        #figs.append(pd.DataFrame(chest_bs_smooth, columns=["Chest Bandstop Smoothed"]).plot(
        #    title="Chest Bandstop Smoothed vs Time"))
        #figs.append(pl.create_spectrogram(chest_bs_smooth, 25, "Chest Bandstop Smoothed - Spectrogram"))

        pl.figures_to_html(figs, filename=os.path.join(activity_data_dir, "VE-VT.html"), show=True)
