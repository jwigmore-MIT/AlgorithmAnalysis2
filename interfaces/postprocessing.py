import numpy as np
import pandas as pd

from analysis.data_importing import get_raw_from_dir_path
from external.postprocessing.BR_rVE_RTformat import analyze_v2, read_chest_data, calibration, Filter_breathing_signal, \
    Bandstop_Filter, savgol_filter


def BR_rVE_RTformat_wrapper(uncleaned_data_path):
    # Write docstring
    """
    Using BR_rVE_RTformat, get chest_raw, chest_5hz, chest_bs, chest_bs_smooth, time
    :param uncleaned_data_path:
    :return:
    """

    raw_data = get_raw_from_dir_path(uncleaned_data_path)
    chest_raw, fs, hw_v = read_chest_data(raw_data)
    time = np.array([t / fs / 60 for t in range(len(chest_raw))])  # [min]
    max_amp, baseline, fev1 = calibration(chest_raw)
    chest_5hz = Filter_breathing_signal(chest_raw, fs, 5)
    low, high = 1.5, 5
    chest_bs = Bandstop_Filter(chest_raw, fs, low, high)
    window_length = 15
    chest_bs_smooth = savgol_filter(chest_bs, window_length, 1)
    X_bbyb, ora_br, inst_rVT_raw, inst_rVE_raw, win_rVE_raw, internal_load, internal_load_signal, valley_peak_ALL, in_ex_ratio = analyze_v2(
        chest_raw, fs, hw_v, max_amp, baseline, fev1)

    X_bbyb_df = pd.DataFrame(X_bbyb,
                             columns=['breath-by-breath Time', 'BR Inst', 'BR ORA', 'TV Inst', 'TV Smoothed', 'VE Inst',
                                      'VE Smoothed', 'In/Ex Ratio Inst', 'In/Ex Ratio Smoothed'])
    X_bbyb_df["breath-by-breath Time"] = X_bbyb_df["breath-by-breath Time"] * 60
    return chest_raw, chest_5hz, chest_bs, chest_bs_smooth, time, X_bbyb_df
