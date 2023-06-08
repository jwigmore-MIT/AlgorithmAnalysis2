import numpy as np
import pandas as pd

from Old.analysis.data_importing import get_raw_from_dir_path
from Old import external as og_post, external as custom_post


def BR_rVE_RTformat_wrapper(uncleaned_data_path):
    # Write docstring
    """
    Using BR_rVE_RTformat, get chest_raw, chest_5hz, chest_bs, chest_bs_smooth, time
    :param uncleaned_data_path:
    :return:
    """

    raw_data = get_raw_from_dir_path(uncleaned_data_path)
    chest_raw, fs, hw_v = og_post.read_chest_data(raw_data)
    time = np.array([t / fs / 60 for t in range(len(chest_raw))])  # [min]
    max_amp, baseline, fev1 = og_post.calibration(chest_raw)
    chest_5hz = og_post.Filter_breathing_signal(chest_raw, fs, 5)
    low, high = 1.5, 5
    chest_bs = og_post.Bandstop_Filter(chest_raw, fs, low, high)
    window_length = 15
    chest_bs_smooth = og_post.savgol_filter(chest_bs, window_length, 1)
    X_bbyb, ora_br, inst_rVT_raw, inst_rVE_raw, win_rVE_raw, internal_load, internal_load_signal, valley_peak_ALL, in_ex_ratio = og_post.analyze_v2(
        chest_raw, fs, hw_v, max_amp, baseline, fev1)

    X_bbyb_df = pd.DataFrame(X_bbyb,
                             columns=['breath-by-breath Time', 'BR Inst', 'BR ORA', 'TV Inst', 'TV Smoothed', 'VE Inst',
                                      'VE Smoothed', 'In/Ex Ratio Inst', 'In/Ex Ratio Smoothed'])
    X_bbyb_df["breath-by-breath Time"] = X_bbyb_df["breath-by-breath Time"] * 60
    return chest_raw, chest_5hz, chest_bs, chest_bs_smooth, time, X_bbyb_df

def BR_rVE_RTformat_wrapper_peak_detection(uncleaned_data_path):
    raw_data = get_raw_from_dir_path(uncleaned_data_path)
    chest_raw, fs, hw_v = og_post.read_chest_data(raw_data)
    time = np.array([t / fs / 60 for t in range(len(chest_raw))])  # [min]
    max_amp, baseline, fev1 = og_post.calibration(chest_raw)
    chest_5hz = og_post.Filter_breathing_signal(chest_raw, fs, 5)
    low, high = 1.5, 5
    chest_bs = og_post.Bandstop_Filter(chest_raw, fs, low, high)
    window_length = 15
    chest_bs_smooth = og_post.savgol_filter(chest_bs, window_length, 1)
    peaks_cal, dic_cal = og_post.find_peaks(chest_bs_smooth[:60 * fs], prominence=25, distance=fs)
    minProm = np.sort(dic_cal['prominences'])[0]  # second lowest
    if minProm > 100:
        minProm = 100
    elif minProm < 25:
        minProm = 25
    # Apply to full record to get indices of peaks and valleys
    peaks, dic = og_post.find_peaks(chest_bs_smooth, prominence=minProm / 2, distance=fs / 1.5) # returns indices and properties on smoothed chest data
    valleys, dic_v = og_post.find_peaks(-chest_bs_smooth, prominence=minProm / 2, distance=fs / 1.5)

    peakTS = time[peaks]*60
    valleyTS = time[valleys]*60

    peakTS = peakTS.reshape(-1, 1)
    valleyTS = valleyTS.reshape(-1, 1)

    # round to nearest 0.01
    peakTS = np.round(peakTS, 2)
    valleyTS = np.round(valleyTS, 2)

    min_index = min(len(peakTS), len(valleyTS))
    # Merge peakTS and valleyTS into one array
    peak_valley_TS = np.concatenate((peakTS[:min_index], valleyTS[:min_index]), axis=1)

    # convert to dataframe
    peak_valley_TS_df = pd.DataFrame(peak_valley_TS, columns=["PeakTS", "ValleyTS"])

    return peak_valley_TS_df

def custom_post_format_wrapper(uncleaned_data_path):
    # Write docstring
    """
    Using BR_rVE_RTformat, get chest_raw, chest_5hz, chest_bs, chest_bs_smooth, time
    :param uncleaned_data_path:
    :return:
    """

    raw_data = get_raw_from_dir_path(uncleaned_data_path)
    chest_raw, fs, hw_v = custom_post.read_chest_data(raw_data)
    time = np.array([t / fs / 60 for t in range(len(chest_raw))])  # [min]
    max_amp, baseline, fev1 = custom_post.calibration(chest_raw)
    chest_5hz = custom_post.Filter_breathing_signal(chest_raw, fs, 5)
    low, high = 1.5, 5
    chest_bs = custom_post.Bandstop_Filter(chest_5hz, fs, low, high)
    window_length = 15
    chest_bs_smooth = custom_post.savgol_filter(chest_bs, window_length, 1)
    X_bbyb, ora_br, inst_rVT_raw, inst_rVE_raw, win_rVE_raw, internal_load, internal_load_signal, valley_peak_ALL, in_ex_ratio = custom_post.analyze_v2(
        chest_raw, fs, hw_v, max_amp, baseline, fev1)

    X_bbyb_df = pd.DataFrame(X_bbyb,
                             columns=['breath-by-breath Time', 'BR Inst', 'BR ORA', 'TV Inst', 'TV Smoothed', 'VE Inst',
                                      'VE Smoothed', 'In/Ex Ratio Inst', 'In/Ex Ratio Smoothed'])
    X_bbyb_df["breath-by-breath Time"] = X_bbyb_df["breath-by-breath Time"] * 60

    return {"chest_raw": chest_raw,
            "chest_5hz": chest_5hz,
            "chest_bs": chest_bs,
            "chest_bs_smooth": chest_bs_smooth,
            "time": time,
            "X_bbyb_df": X_bbyb_df,
            "valley_peak_ALL": valley_peak_ALL}
