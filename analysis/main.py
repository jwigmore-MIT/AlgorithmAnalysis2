import os

import pandas as pd

pd.options.plotting.backend = "plotly"
import data_importing as imp  # Custom importing module
import plotting as pl  # Custom plotting module
import interfaces.postprocessing as pif  # post processing interface

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # activity_data_dir = "User_data/Arnar_Larusson_2023-04-12_easy_ride"
    # activity_data_dir = "User_data/Juan_2023-03-18_Testing_Live_VE_and_Garmin"
    activity_data_dir = "../data/Juan_2023-03-18_Testing_Live_VE_and_Garmin"
    uncleaned_data_dir = os.path.join(activity_data_dir, "uncleaned_data")
    cleaned_data_dir = os.path.join(activity_data_dir, "cleaned_data")
    imp.clean_all_data(uncleaned_data_dir)
    clean_dfs = imp.load_cleaned_data(cleaned_data_dir)

    if True:
        live_algo_sz = clean_dfs["live_b3_df"].size
        raw_slow_sz = clean_dfs["raw_slow_df"].size
        raw_fast_sz = clean_dfs["raw_fast_df"].size
        device_upload_sz = live_algo_sz + raw_fast_sz + raw_slow_sz
        live_frac = live_algo_sz / device_upload_sz
        slow_frac = raw_slow_sz / device_upload_sz
        fast_frac = raw_fast_sz / device_upload_sz
        print("AS IS")
        print(f"125Hz Sample Rate Data Fraction = {fast_frac}")
        print(f"25Hz Sample Rate Data Fraction = {slow_frac}")
        print(f"Live Algo Data Fraction = {live_frac}")

        elim_speed_up = (raw_slow_sz + live_algo_sz) / device_upload_sz
        live_red_speed_up = (raw_slow_sz + live_algo_sz + raw_fast_sz / 5) / device_upload_sz
        print(f"Relative upload time percentage if we eliminated the 125Hz data {elim_speed_up}")
        print(f"Relative upload time percentage if we reduced 125Hz to 25Hz samples {live_red_speed_up}")

    raw_slow_df = clean_dfs["raw_slow_df"]
    aws_b3_df = clean_dfs["aws_b3_df"]
    live_b3_df = clean_dfs["live_b3_df"]

    # Post Processing Data from BR_rVE_RTformat
    chest_raw, chest_5hz, chest_bs, chest_bs_smooth, time, X_bbyb_df = pif.BR_rVE_RTformat_wrapper(uncleaned_data_dir)

    # raw_fig_dict = {
    #     "Raw Chest": (raw_slow_df, "time", "c")
    # }
    #
    # VE_fig_dict = {
    #     "Post Processing": (aws_b3_df, "breathTime", "VE"),
    #     "Live": (live_b3_df, "breathTime", "VE"),
    # }
    #
    # pl.create_subplots_w_raw(VE_fig_dict, raw_fig_dict,xtitle= "Time [s]", ytitle1= "VE", ytitle2="Raw Chest" ,show= True)
    #
    # VT_fig_dict = {
    #     "Post Processing": (aws_b3_df, "breathTime", "VT"),
    #     "Live": (live_b3_df, "breathTime", "VT"),
    # }
    # pl.create_subplots_w_raw(VT_fig_dict,raw_fig_dict, xtitle = "Time [s]", ytitle1= "VT", ytitle2= "Raw Chest", show = True)
    if True:
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
