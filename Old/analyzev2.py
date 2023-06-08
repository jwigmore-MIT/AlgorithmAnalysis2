def analyze_v2(chest_raw, fs, hw_v, max_amp, baseline, fev1):
    '''
    folder = '/Users/gregorylaredo/Desktop/TymeWear/data/test'#Beta Users'
    os.chdir(folder)
    json_file = '2021-07-21_workout_1_Raw.json'
    fs=25
    json_data = json.loads(open(json_file).read())
    json_info = json_data['info']
    '''
    time = np.array([t / fs / 60 for t in range(len(chest_raw))])  # [min]

    if np.all((chest_raw == 0)):
        print("Breathing signal error: No signal detected")  # if no breaths were detected can't continue
        ora_br = []
        inst_rVT_raw = []
        inst_rVE_raw = []
        win_rVE_raw = []
        internal_load = []
        internal_load_signal = []
        valley_peak_ALL = np.empty((0, 2))
        in_ex_ratio = np.empty((0, 2))
        X_bbyb = []
    else:

        # 0) Show full spectrum
        chest_5hz = Filter_breathing_signal(chest_raw, fs, 5)
        w = 15
        chest_bs_smooth = savgol_filter(chest_bs, w, 1)

        # 3) peak and valley detection
        # Determine min prom from 60sec Calibration
        peaks_cal, dic_cal = find_peaks(chest_bs_smooth[:60 * fs], prominence=25, distance=fs)
        minProm = np.sort(dic_cal['prominences'])[0]  # second lowest
        if minProm > 100:
            minProm = 100
        elif minProm < 25:
            minProm = 25
        # Apply to full record
        peaks, dic = find_peaks(chest_bs_smooth, prominence=minProm / 2, distance=fs / 1.5) # returns indices and properties
        valleys, dic_v = find_peaks(-chest_bs_smooth, prominence=minProm / 2, distance=fs / 1.5)


        i = 0  # peaks ind
        j = 0  # valleys ind
        valley_peak_ALL = []
        # First remove multiple valleys in front of ending peak:
        while i < len(peaks):
            prev_js = np.argwhere(valleys < peaks[i])
            if prev_js.size == 0:
                i = i + 1
            else:
                j = prev_js[-1][0]  # take closest valley to proceeding peak
                valley_peak_ALL.append([valleys[j], peaks[i]])
                i = i + 1
                j = j + 1
        valley_peak_ALL = np.array(valley_peak_ALL)
        # Then remove multiple peaks after same startig valley:
        del_is = np.argwhere(np.diff(valley_peak_ALL[:, 0]) == 0)
        valley_peak_ALL = np.delete(valley_peak_ALL, del_is, 0)
        valley_peak_Smoothed_ALL = np.transpose(
            np.array([chest_bs_smooth[valley_peak_ALL[:, 0]], chest_bs_smooth[valley_peak_ALL[:, 1]]]))
        # Only keep breaths detected where the peak detected is larger than the proceeding valley detected
        a = np.argwhere(valley_peak_Smoothed_ALL[:, 1] > valley_peak_Smoothed_ALL[:, 0])[:, 0]
        valley_peak_ALL = valley_peak_ALL[a, :]
        valley_peak_Smoothed_ALL = valley_peak_Smoothed_ALL[a, :]

        # PEAK=TO-PEAK
        # 3) Breath-by-breath Inst + Outlier-rejection Avg BR Calculations
        inst_br = fs * 60 / np.diff(valley_peak_ALL[:, 1])
        inst_br_ind = valley_peak_ALL[1:, 1]
        inst_br_time = inst_br_ind / fs / 60  # [min]

        br_win = 5
        br_reject = 2
        br = []
        k = 0
        while k <= len(inst_br) - br_win:
            x = inst_br[k:k + br_win]
            mean_x = np.mean(x)
            y = np.abs(x - mean_x)
            a = np.argsort(y)
            keep = a[:-br_reject]
            avg_br = np.mean(x[keep])
            br.append(avg_br)
            k = k + 1
        br = np.round(np.array(br, dtype=np.float), decimals=1)  # [brpm]
        br_time = valley_peak_ALL[br_win:, 1] / fs / 60  # [min]

        # 4) Breath-by-breath Inst + Outlier-rejection Avg rVE -&- 15sec window rVE Calculations

        # ---------------------------------------------------------------
        # Extracting Breath Amplitude (delta_amp) from Raw Breathing Signal instead of Filtered
        # Searching X seconds before and after peak, where X is dependant on BR
        # Skipping the step of defining variable valley_peak_RANGE_raw_ALL
        delta_amp_raw_og = []
        raw_valley = []
        raw_peak = []

        for i in range(1, len(inst_br) + 1):
            x = 60 / inst_br[
                i - 1]  # breath-to-breath duration [sec] of the i-th breath (i.e. i-th minus (i-1)-th peak-to-peak rate)
            x = int(np.ceil(
                (1 / 12) * x * fs))  # search +/- 1/12 of the period length (i.e. pie/6), then convert to [sample no.]
            raw_value_valley = np.mean(
                chest_raw[np.arange(valley_peak_ALL[i, 0] - x, valley_peak_ALL[i, 0] + x, 1, dtype=int)])
            raw_value_peak = np.mean(chest_raw[np.arange(valley_peak_ALL[i, 1] - x, valley_peak_ALL[i, 1] + x, 1)])
            delta_amp_raw_og.append(raw_value_peak - raw_value_valley)
            raw_valley.append(raw_value_valley)
            raw_peak.append(raw_value_peak)
        delta_amp_raw_og = np.array(delta_amp_raw_og, dtype=np.float)
        raw_valley = np.array(raw_valley, dtype=np.float)
        raw_peak = np.array(raw_peak, dtype=np.float)
        keep_og = np.argwhere((delta_amp_raw_og >= 0))[:, 0]
        delta_amp_raw = delta_amp_raw_og[keep_og]
        inst_br_temp = inst_br[keep_og]
        inst_br_ind_temp = inst_br_ind[keep_og]
        inst_rVE_raw = []
        for k in range(0, len(inst_br_temp)):
            inst_rVE_raw.append(inst_br_temp[k] * delta_amp_raw[k] / 100)
        inst_rVE_raw = np.array(inst_rVE_raw, dtype=np.float)
        inst_rVE_time_raw = inst_br_ind_temp / fs / 60  # [min]
        # ---------------------------------------------------------------

        delta_amp = []
        for k in range(1, len(inst_br) + 1):
            delta_amp.append(valley_peak_Smoothed_ALL[k, 1] - valley_peak_Smoothed_ALL[k, 0])
        delta_amp = np.array(np.sqrt(delta_amp), dtype=np.float)
        inst_rVE = []
        for k in range(0, len(inst_br)):
            inst_rVE.append(inst_br[k] * delta_amp[k] / 100)
        inst_rVE = np.array(inst_rVE, dtype=np.float)

        # Windowed rVE based on inst. VE (original):
        win = 15 * fs
        slide = 1 * fs
        rVE = []
        k = 0
        while k < inst_br_ind[-1] - np.mod(inst_br_ind[-1], fs) - win + slide:
            keep = np.argwhere((inst_br_ind >= k) & (inst_br_ind < k + win))
            rVE.append(np.sum(inst_rVE[keep]) / 10 * 60 / (win / fs))
            k = k + slide
        rVE = np.array(rVE, dtype=np.float)

        # Windowed rVE based on inst. Raw VE:
        win = 15 * fs
        slide = 1 * fs
        rVE_raw = []
        k = 0
        while k < inst_br_ind_temp[-1] - np.mod(inst_br_ind_temp[-1], fs) - win + slide:
            keep = np.argwhere((inst_br_ind_temp >= k) & (inst_br_ind_temp < k + win))
            rVE_raw.append(np.sum(inst_rVE_raw[keep]) / 10 * 60 / (win / fs))
            k = k + slide
        rVE_raw = np.array(rVE_raw, dtype=np.float)
        rVE_raw_time = np.linspace(win / fs / 60, ((k + win) / fs - 1) / 60, int(k / fs))  # [min]

        # Smoothing
        if len(br) >= 31:
            br_smooth = savgol_filter(br, 31, 1)
        else:
            br_smooth = br
        if len(rVE_raw) >= 61:
            rVE_raw_smooth = savgol_filter(rVE_raw, 61, 1)
        else:
            rVE_raw_smooth = rVE_raw
        if len(inst_rVE_raw) >= 21:
            inst_rVE_smooth_raw = savgol_filter(inst_rVE_raw, 21, 1)
            delta_amp_raw_smooth = savgol_filter(delta_amp_raw, 21, 1)
        else:
            inst_rVE_smooth_raw = inst_rVE_raw
            delta_amp_raw_smooth = delta_amp_raw

        # PEAK-TO-PEAK:
        ora_br = np.transpose(np.vstack((br_time, br, br_smooth)))
        inst_rVT_raw = np.transpose(np.vstack((inst_rVE_time_raw, delta_amp_raw, delta_amp_raw_smooth,
                                               delta_amp_raw / fev1, delta_amp_raw_smooth / fev1)))
        inst_rVE_raw = np.transpose(np.vstack((inst_rVE_time_raw, inst_rVE_raw, inst_rVE_smooth_raw)))
        win_rVE_raw = np.transpose(np.vstack((rVE_raw_time, rVE_raw, rVE_raw_smooth)))

        # Internal Load
        ve = win_rVE_raw  # inst_rVE_raw # VE_baseline_multiple ## <<< Select VE signal to use
        ve_sig = ve[:, 2]  # ve[:,1] ##- VE_baseline_thresh
        ve_sig[ve_sig < 0] = 0
        internal_load = sum(ve_sig) / 1000  # *2 #/100
        internal_load_signal = np.cumsum(ve_sig * 10)
        internal_load_signal = np.transpose(np.vstack((ve[:, 0], internal_load_signal)))

        # Inhale:Exhale duration ratio
        in_ = valley_peak_ALL[1:, 1] - valley_peak_ALL[1:, 0]  # peak(i) - valley(i) ## valley_1 >>> peak_1
        ex_ = valley_peak_ALL[1:, 0] - valley_peak_ALL[:-1, 1]  # valley(i+1) - peak(i)  ## peak_1 >>> valley_2
        in_ex = in_[1:] / ex_[1:]
        if len(in_ex) >= 15:
            in_ex_smooth = savgol_filter(in_ex, 15, 1)
        else:
            in_ex_smooth = in_ex
        in_ex_ratio = np.transpose(np.vstack((inst_br_time[:-1], in_ex, in_ex_smooth)))

        X_bbyb = np.zeros((len(valley_peak_ALL[:, 1]), 9))
        '''
        X_bbyb[:,0] = breath-by-breath Time [sec]
        X_bbyb[:,1] = BR Inst.
        X_bbyb[:,2] = BR ORA (3/5)
        X_bbyb[:,3] = TV Inst. 
        X_bbyb[:,4] = TV Smoothed (savgol 21)
        X_bbyb[:,5] = VE Inst. 
        X_bbyb[:,6] = VE Smoothed (savgol 21)
        X_bbyb[:,7] = In/Ex Ratio Inst. 
        X_bbyb[:,8] = In/Ex Ratio Smoothed (savgol 15)
        '''
        X_bbyb[:] = np.nan
        X_bbyb[:, 0] = np.round(valley_peak_ALL[:, 1] / 60 / fs, 4)
        X_bbyb[1:, 1] = np.round(inst_br, 2)
        X_bbyb[br_win:, 2] = br
        X_bbyb[keep_og + 1, 3] = np.round(delta_amp_raw, 1)
        X_bbyb[keep_og + 1, 4] = np.round(delta_amp_raw_smooth, 1)
        X_bbyb[keep_og + 1, 5] = np.round(inst_rVE_raw[:, 1], 1)
        X_bbyb[keep_og + 1, 6] = np.round(inst_rVE_smooth_raw, 1)
        X_bbyb[1:-1, 7] = np.round(in_ex, 3)
        X_bbyb[1:-1, 8] = np.round(in_ex_smooth, 3)



    return X_bbyb, ora_br, inst_rVT_raw, inst_rVE_raw, win_rVE_raw, internal_load, internal_load_signal, valley_peak_ALL, in_ex_ratio