"""
Date: Sep 25, 2020
Author: Gregory Laredo

DESCRIPTION:
Extracts breath-by-breath BR and rVE calculation algorithm via sliding-window 
signal processing (i.e. the real-time format algorithm) enabling adaptive 
filterring and adaptive breath detection.
"""

# from scipy.fftpack import fft
# from scipy.fftpack import fft
import warnings

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, periodogram, savgol_filter, spectrogram  # , welch

warnings.filterwarnings("ignore")


def read_chest_data(json_data, fs=50, fs_25=False, interp=True):
    data = json_data['samples']

    if fs_25:
        data_25 = []
        i = 0
        while i < len(data):
            data_25.append(data[i])
            i += 2
        data = data_25

    if 'missing-points' in json_data:
        points = json_data['missing-points']
        samples = json_data['samples']
        missing_points = {point['id']: point for point in points}
        samples = [sample for sample in samples if sample['id'] not in missing_points]
        for point_id, point in missing_points.items():
            samples.insert(point_id, point)
        data = samples

    try:
        chest_raw = np.array([d['chest'] for d in data])
        id_ = np.array([d['id'] for d in json_data['samples']])
    except:
        chest_raw = np.array([d['c'] for d in data])
        id_ = np.array([d['id'] for d in json_data['samples']])
    try:
        fs = json_data['info']['fs']
    except:
        fs = fs
    try:
        hw_v = int(json_data['info']['hw_v'])  # 1 = old, 2 = parker, 3 = parker also
    except:
        hw_v = int(0)

    if interp:
        if np.sum(np.diff(id_)) + 1 > len(chest_raw):
            sample_true = id_ - id_[0]
            sample_interp = np.arange(0, sample_true[-1] + 1)
            chest_raw = np.array(np.interp(sample_interp, sample_true, chest_raw), dtype=int)
            id_ = np.array(np.interp(sample_interp, sample_true, id_), dtype=int)
        elif np.sum(np.diff(id_)) + 1 < len(chest_raw):
            if len(json_data['info']['duration']) == (5 or 7):
                dur = mmss2min(json_data['info']['duration'])
            else:  # len(json_data['info']['duration']) == (8 or 10):
                dur = hhmmss2min(json_data['info']['duration'])
            #            temp = np.zeros((int(dur*60*fs - len(chest_raw)),1))[:,0]
            #            temp[:] = np.nan\
            if dur * 60 * fs > len(chest_raw):
                id_change = np.argwhere(np.diff(id_) != 1)[:, 0]
                for i in range(len(id_change)):
                    id_add = np.arange(id_[id_change[i]] + 1, id_[id_change[i] + 1])
                    if id_change[i] < fs * 60:
                        chest_raw_add = np.linspace(chest_raw[id_change[i]], chest_raw[id_change[i] + 1],
                                                    num=len(id_add) + 2, dtype=int)[1:-1]
                    else:
                        chest_raw_add = np.zeros((int(dur * 60 * fs - len(chest_raw)), 1))[:, 0]
                        chest_raw_add[:] = np.nan
                    chest_raw = np.hstack((chest_raw[:id_change[i] + 1], chest_raw_add, chest_raw[id_change[i] + 1:]))
                    id_ = np.hstack((id_[:id_change[i] + 1], id_add, id_[id_change[i] + 1:]))
                    id_change = id_change + len(id_add)
    #            chest_raw = np.hstack((chest_raw[:id_change],temp,chest_raw[id_change:]))

    if 'pause_resume_time_stamps' in json_data['info'] and json_data['info']['pause_resume_time_stamps'] != []:
        start_epoch = json_data['info']['timestamp']
        id_epoch = start_epoch + id_ / fs
        pause_resume_epochs = np.array(json_data['info']['pause_resume_time_stamps'],
                                       dtype=float) / 1000  # its in epoch to the millisecond (i.e. has 3 ectra places)
        pause_resume_id = np.array(np.round(fs * (pause_resume_epochs - id_epoch[0]), 0), dtype=int)
        # trim if odd length of 'pause-resume' entries because don't need last 'pause' since its same as 'end'
        if np.mod(len(pause_resume_id), 2) == 1:
            pause_resume_id = pause_resume_id[:-1]
        if len(pause_resume_id) >= 2:
            nan_add = pause_resume_id[1::2] - pause_resume_id[::2]
            chest_raw_NEW = np.zeros((len(chest_raw) + np.sum(nan_add)))
            chest_raw_NEW[:] = np.nan
            # initiate until first 'pause'
            chest_raw_NEW[:pause_resume_id[0]] = chest_raw[:pause_resume_id[0]]
            # go through all susequent 'resumes' and 'pauses'
            for i in range(0, len(pause_resume_id), 2):
                if i + 2 < len(pause_resume_id):
                    chest_raw_NEW[
                    pause_resume_id[i] + np.sum(nan_add[:int(i / 2) + 1]) + 1:pause_resume_id[i + 2] + np.sum(
                        nan_add[:int(i / 2) + 1])] = chest_raw[pause_resume_id[i] + 1:pause_resume_id[i + 2]]
                else:
                    chest_raw_NEW[pause_resume_id[i] + np.sum(nan_add[:int(i / 2) + 1]) + 1:] = chest_raw[
                                                                                                pause_resume_id[i] + 1:]
            chest_raw = chest_raw_NEW[:]

    if fs_25:
        fs = fs // 2

    return chest_raw, fs, hw_v


def hhmmss2min(hhmmss):
    '''
    hhmmss = ['hh:mm:ss']
    '''
    hhmmss = np.fromstring(hhmmss, sep=':')
    if len(hhmmss) == 3:
        mins = hhmmss[0] * 60 + hhmmss[1] + hhmmss[2] / 60
    elif len(hhmmss) == 2:
        mins = hhmmss[0] + hhmmss[1] / 60
    return mins


def mmss2min(mmss):
    '''
    mmss = ['mm:ss']
    '''
    mmss = np.fromstring(mmss, sep=':')
    mins = mmss[0] + mmss[1] / 60
    return mins


def Filter_breathing_signal(signal, fs, highcut, lowcut=0.1, order=2):
    signal = signal - np.nanmean(signal)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    signal_filt = filtfilt(b, a, signal, padlen=np.int(0.5 * len(signal)))
    return signal_filt


def Bandstop_Filter(signal, fs, lowcut, highcut, order=2):
    signal = signal - np.nanmean(signal)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    signal_filt = filtfilt(b, a, signal, padlen=np.int(0.5 * len(signal)))
    return signal_filt


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
        #        # Testing:
        #        fs_ = 1000
        #        x = np.arange(0,100,1/fs_)
        ##        ss = 5
        ##        se = 10
        ##        y = np.arange(ss,se,(se-ss)/len(x))
        #        y = 8
        #        y = y*np.sin(2*x)
        #        f, t, Sxx = spectrogram(y, fs_,scaling='spectrum',mode='psd')
        #
        #        fp, Pxx = periodogram(y,fs,scaling='spectrum')
        #
        #        import matplotlib.pyplot as plt
        #
        #        plt.figure(figsize=(6,3))
        #        plt.plot(x,y)
        #        plt.ylabel('Signal')
        #        plt.xlabel('Time [sec]')
        #
        #        plt.figure(figsize=(6,3))
        #        plt.plot(fp,Pxx)
        #        plt.xlabel('Frequency [Hz]')
        #        plt.ylabel('Power')
        #        plt.xlim(0,2)
        ##        plt.ylim(0,500)
        #
        #        plt.figure(figsize=(6,3))
        #        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        #        plt.ylabel('Frequency [Hz]')
        #        plt.xlabel('Time [sec]')
        #        plt.ylim(0,1)
        #        plt.xlim(0,x[-1])
        #        plt.show()
        # - - - - - - - - - - - - - - - - -

        # 0) Show full spectrum
        chest_5hz = Filter_breathing_signal(chest_raw, fs, 5)
        f, t, Sxx = spectrogram(chest_5hz, fs)
        #        # ========================
        #        import matplotlib.pyplot as plt
        #        plt.figure(figsize=(12,6))
        #        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        #        plt.ylabel('Frequency [Hz]')
        #        plt.xlabel('Time [sec]')
        #        plt.ylim(0,5)
        #        plt.xlim(0,time[-1]*60)
        #        plt.show()
        #        # ========================

        # 1) Bandstop filter cadence frequency range
        low, high = 1.5, 5
        chest_bs = Bandstop_Filter(chest_raw, fs, low, high)
        f, t, Sxx = spectrogram(chest_bs, fs)

        #        # ========================
        #        import matplotlib.pyplot as plt
        #        plt.figure(figsize=(12,6))
        #        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        #        plt.ylabel('Frequency [Hz]')
        #        plt.xlabel('Time [sec]')
        #        plt.ylim(0,5)
        #        plt.xlim(0,time[-1]*60)
        #        plt.show()
        ##
        ##        fig, axs = plt.subplots(3, 1, figsize=(12,9), sharex=True)
        ##        axs[0].plot(time*60,chest_raw,'k-',lw=1)
        ##        axs[0].set_ylabel('Raw')
        ##        axs[0].set_xlim(0,time[-1]*60)
        ##        axs[0].grid(True)
        ##        #--------------
        ##        chest_2hz = Filter_breathing_signal(chest_raw,fs,5)
        ##        axs[1].plot(time*60,chest_2hz,'b-',lw=1)
        ##        axs[1].set_ylabel('Raw: BP 0.3-2Hz')
        ##        axs[1].set_xlim(0,time[-1]*60)
        ##        axs[1].grid(True)
        ##        #--------------
        ##        axs[2].plot(time*60,chest_bs,'g-',lw=1)
        ##        axs[2].set_ylabel('Raw: BS %.1f-%.1fHz' %(low,high))
        ##        axs[2].set_xlim(0,time[-1]*60)
        ##        axs[2].grid(True)
        ##        axs[2].set_xlim(600,650)
        ##        # ========================

        # 2) Savgol smoothing
        #        # ---- Test for best win size ----
        #        no = 4
        #        chest_bs_smooth_all = np.zeros((len(chest_bs),no))
        #        w = 4
        #        for i in range(no):
        #            chest_bs_smooth_all[:,i] = savgol_filter(chest_bs,w+1,1)
        #            w = w*2
        #
        #        fig, axs = plt.subplots(no+1, 1, figsize=(12,(no+1)*3), sharex=True, sharey=True)
        #        axs[0].plot(time*60,chest_bs,'g-',lw=1)
        #        axs[0].set_ylabel('Raw: BS %.1f-%.1fHz' %(low,high))
        #        axs[0].grid(True)
        #        c = 0.3
        #        w = w/(2*no)/2
        #        for i in range(no):
        #            axs[i+1].plot(time*60,chest_bs_smooth_all[:,i],'-',color=[c,c,c],lw=1,label=('win = %i' %(w+1)))
        #            axs[i+1].set_ylabel('Savgol Smoothed')
        #            axs[i+1].legend(loc='upper left',fontsize=10)
        #            axs[i+1].grid(True)
        #            c = c+0.1
        #            w = w*2
        #        axs[i].set_xlim(600,650)
        ##        axs[i].set_xlim(0,time[-1]*60)
        #        plt.show()
        #        # --------------------------------
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
        peaks, dic = find_peaks(chest_bs_smooth, prominence=minProm / 2, distance=fs / 1.5)
        valleys, dic_v = find_peaks(-chest_bs_smooth, prominence=minProm / 2, distance=fs / 1.5)

        #        # ========================
        #        import matplotlib.pyplot as plt
        #        fig, axs = plt.subplots(3, 1, figsize=(12,9), sharex=True)
        #        axs[0].plot(time*60,chest_raw-np.mean(chest_raw),'k-',lw=1)
        #        axs[0].set_ylabel('Raw')
        #        axs[0].grid(True)
        #        #--------------
        #        axs[1].plot(time*60,chest_bs,'g-',lw=1)
        #        axs[1].set_ylabel('BS %.1f-%.1fHz' %(low,high))
        #        axs[1].set_xlim(0,time[-1]*60)
        #        axs[1].grid(True)
        #        #--------------
        #        axs[2].plot(time*60,chest_bs_smooth,'-',color=[0.3,0.3,0.3],lw=1)
        #        axs[2].plot(time[peaks]*60,chest_bs_smooth[peaks],'r.',alpha=0.5,markersize=14)
        #        axs[2].plot(time[valleys]*60,chest_bs_smooth[valleys],'b.',alpha=0.5,markersize=14)
        #        axs[2].set_ylabel('Sav-Gol (win=%i)' % (w))
        #        axs[2].set_xlim(0,time[-1]*60)
        #        axs[2].grid(True)
        #        axs[2].set_xlim(0,100)#600,650)
        #        # ========================

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

    #        # ========================
    #        fig, axs = plt.subplots(3, 1, figsize=(12,9), sharex=True)
    #        fig.subplots_adjust(hspace=0.1)
    #        plt.suptitle('%s' % (json_file),fontsize=14,y=0.97)
    #        i = 0
    #        axs[i].plot(time,chest_raw,'k-',linewidth=0.75,alpha=0.5)
    #        axs[i].hold(True)
    #        axs[i].plot(time[valley_peak_ALL[1:,0]],raw_valley,'r.',alpha=1)
    #        axs[i].plot(time[valley_peak_ALL[1:,1]],raw_peak,'g.',alpha=1)
    #        #axs[i].plot(time[valley_peak_ALL[1:,0]],chest_raw[valley_peak_ALL[1:,0]],'r.',alpha=0.75)
    #        #axs[i].plot(time[valley_peak_ALL[1:,1]],chest_raw[valley_peak_ALL[1:,1]],'g.',alpha=0.75)
    #        axs[i].legend(('signal','raw valley (avg.)','raw peak (avg.)'), loc='upper left',fontsize=10)
    #        axs[i].set_ylabel('Raw Breathing Signal',fontsize=12)
    #        axs[i].grid(True)
    #        axs[i].set_xlim(0,time[-1])
    #        #     i = i + 1
    #        #     axs[i].plot(time,chest_bs,'b-',linewidth=1,alpha=0.75)
    #        #     axs[i].hold(True)
    #        #     axs[i].plot(time[valley_peak_ALL[1:,0]],chest_bs[valley_peak_ALL[1:,0]],'r.',alpha=1)
    #        #     axs[i].plot(time[valley_peak_ALL[1:,1]],chest_bs[valley_peak_ALL[1:,1]],'g.',alpha=1)
    #        #     axs[i].legend(('signal','filt valley','filt peak'), loc='upper left',fontsize=10)
    #        #     axs[i].set_ylabel('BS Filterred Breathing Signal',fontsize=12)
    #        #     axs[i].grid(True)
    #        #     axs[i].set_xlim(0,time[-1])
    #        i = i + 1
    #        axs[i].plot(time,chest_bs_smooth,'b-',linewidth=1,alpha=0.75)
    #        axs[i].hold(True)
    #        axs[i].plot(time[valley_peak_ALL[1:,0]],chest_bs_smooth[valley_peak_ALL[1:,0]],'r.',alpha=1)
    #        axs[i].plot(time[valley_peak_ALL[1:,1]],chest_bs_smooth[valley_peak_ALL[1:,1]],'g.',alpha=1)
    #        axs[i].legend(('signal','smooth valley','smooth peak'), loc='upper left',fontsize=10)
    #        axs[i].set_ylabel('BS Filterred + Smoothed Breathing Signal',fontsize=12)
    #        axs[i].grid(True)
    #        axs[i].set_xlim(0,time[-1])
    #        i = i + 1
    #        axs[i].plot(br_time,br,'m.',alpha=0.2)
    #        axs[i].hold(True)
    #        axs[i].plot(br_time,br_smooth,'m-',linewidth=1.5,alpha=1)
    #        #axs[i].plot(time_imu,az,'g-',linewidth=1,alpha=0.75)
    #        axs[i].set_ylabel('Breathing Rate',fontsize=12)
    #        axs[i].legend(('inst.','smoothed'),fontsize=10,loc='upper left')
    #        axs[i].grid(True)
    #        axs[i].set_xlim(0,time[-1])
    #        axs[i].set_xlabel('Time [$sec$]',fontsize=12)
    #        plt.show()
    #        # ========================

    return X_bbyb, ora_br, inst_rVT_raw, inst_rVE_raw, win_rVE_raw, internal_load, internal_load_signal, valley_peak_ALL, in_ex_ratio


def calibration(json_data, fs=25):
    try:
        chest_raw, fs, hw_v = read_chest_data(json_data)
        low, high = 1.5, 5
        chest_bs = Bandstop_Filter(chest_raw, fs, low, high)
        f, t, Sxx = spectrogram(chest_bs, fs)
        w = 15
        chest_bs_smooth = savgol_filter(chest_bs, w, 1)
        # Determine min prom from 60sec Calibration
        peaks_cal, dic_cal = find_peaks(chest_bs_smooth[:60 * fs], prominence=25, distance=fs)
        minProm = np.sort(dic_cal['prominences'])[0]  # second lowest
        if minProm > 100:
            minProm = 100
        elif minProm < 25:
            minProm = 25
        # Apply to full record
        peaks, dic = find_peaks(chest_bs_smooth, prominence=minProm, distance=fs / 1.5)
        valleys, dic_v = find_peaks(-chest_bs_smooth, prominence=minProm, distance=fs / 1.5)
        i = 0  # peaks ind
        j = 0  # valleys ind
        peak_valley_ALL = []
        # First remove multiple valleys in front of ending peak:
        while i < len(peaks):
            next_js = np.argwhere(peaks[i] < valleys)
            if next_js.size == 0:
                if i == len(peaks) - 1:
                    last_v = np.argmin(chest_bs_smooth[peaks[i]:peaks[i] + 2 * fs])
                    peak_valley_ALL.append([peaks[i], peaks[i] + last_v])
                i = i + 1
            else:
                j = next_js[0][0]  # take closest valley to proceeding peak
                peak_valley_ALL.append([peaks[i], valleys[j]])
                i = i + 1
                j = j + 1
        #            print('\ni=%i'%i)
        #            print('j=%i'%j)
        #            print(peak_valley_ALL[-1])
        peak_valley_ALL = np.array(peak_valley_ALL)
        # Then remove multiple peaks after same ending valley:
        del_is = np.argwhere(np.diff(peak_valley_ALL[:, 1]) == 0)
        peak_valley_ALL = np.delete(peak_valley_ALL, del_is, 0)
        peak_valley_Smoothed_ALL = np.transpose(
            np.array([chest_bs_smooth[peak_valley_ALL[:, 0]], chest_bs_smooth[peak_valley_ALL[:, 1]]]))
        # Only keep breaths detected where the peak detected is larger than the proceeding valley detected
        a = np.argwhere(peak_valley_Smoothed_ALL[:, 0] > peak_valley_Smoothed_ALL[:, 1])[:, 0]
        peak_valley_ALL = peak_valley_ALL[a, :]
        peak_valley_Smoothed_ALL = peak_valley_Smoothed_ALL[a, :]

        delta_amp_raw = np.round(-np.diff(chest_raw[peak_valley_ALL]), 0)
        delta_amp_smooth = np.round(-np.diff(chest_bs_smooth[peak_valley_ALL]), 0)
        ind_max = peak_valley_ALL[np.argmax(delta_amp_raw), :]

        amp = []
        for i in range(0, len(chest_bs_smooth) - fs):
            amp.append(chest_bs_smooth[i] - chest_bs_smooth[i + (fs)])
        fev1 = np.max(amp)
        ind = amp.index(fev1)

        # ## ----------------------------------------
        # import matplotlib.pyplot as plt

        # fig = plt.figure(figsize=(15,7.5))
        # plt.subplot(311)
        # plt.plot(chest_raw)
        # plt.plot(peak_valley_ALL[:,1], chest_raw[peak_valley_ALL[:,1]],'go')
        # plt.plot(peak_valley_ALL[:,0], chest_raw[peak_valley_ALL[:,0]],'ro')
        # plt.ylabel('Raw',fontsize=12)
        # plt.grid(True)

        # plt.subplot(312)
        # plt.plot(chest_bs)
        # plt.plot(peak_valley_ALL[:,1], chest_bs[peak_valley_ALL[:,1]],'go')
        # plt.plot(peak_valley_ALL[:,0], chest_bs[peak_valley_ALL[:,0]],'ro')
        # plt.ylabel(('BS Filter (%.1f-%iHz)'%(low,high)),fontsize=12)
        # plt.grid(True)

        # plt.subplot(313)
        # plt.plot(chest_bs_smooth)
        # plt.plot(peak_valley_ALL[:,1], chest_bs_smooth[peak_valley_ALL[:,1]],'ro')
        # plt.plot(peak_valley_ALL[:,0], chest_bs_smooth[peak_valley_ALL[:,0]],'go')
        # plt.plot(range(ind_max[0],ind_max[1]),chest_bs_smooth[ind_max[0]:ind_max[1]], c='c',alpha=0.35,lw=7,label='MaxAmp')
        # plt.plot(range(ind,ind+fs),chest_bs_smooth[ind:ind+fs], c='orange',alpha=0.5,lw=5,label='FEV1')
        # plt.ylabel(('Smoothed (win=%i)'%(w)),fontsize=12)
        # plt.xlabel(('Sample No. (Fs=%iHz)'%(fs)),fontsize=12)
        # plt.legend(loc='upper right',fontsize=10)
        # plt.grid(True)

        # plt.show()
        # ## ----------------------------------------

        # Outputs:
        max_amp = int(np.max(delta_amp_raw))
        baseline = int(np.round(np.mean(chest_raw), 0))
        fev1 = int(np.round(fev1, 0))

    except:
        max_amp = np.nan
        baseline = np.nan
        fev1 = np.nan

    return max_amp, baseline, fev1


def analyze(chest_raw, fs, hw_v, max_amp, baseline, fev1):  # (json_data, max_amp, baseline, fev1, fs=50):
    # --------------------------------
    #    import os
    #    os.chdir('/Users/gregorylaredo/Desktop/TymeWear/Algorithms/data/Race Mania Expo 2019')
    #    json_file = 'Michael Weintraub (2).json' # 'Chas Hodgdon.json' #
    #
    ##    os.chdir('/Users/gregorylaredo/Desktop/TymeWear/Algorithms/data/Threshold Test/50hz')
    ##    json_file = 'Paul Lang VT 2-26-19.json'
    #
    #    fs = 50
    #    json_data = json.loads(open(tyme_data).read())

    #    fs = 50
    #    json_data = json.loads(open(json_file).read())

    # --------------------------------
    try:
        # 0) Read in JSON file and define Sliding-window Parameters
        # chest_raw, fs, hw_v = read_chest_data(json_data)
        time = np.array([t / fs / 60 for t in range(len(chest_raw))])  # [min]
        if np.all((chest_raw == 0)) or np.all((np.diff(chest_raw[fs:-fs]) == 0)):
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
            s = 3 * fs  # [sample no.] sliding size
            cal = 60 * fs  # [sample no.] calibration size
            w = 18 * fs  # [sample no.] window size
            cut = 3 * fs  # [sample no.] size of start and end of window to cut and not use peak detection results towards BR/rVE (i.e. contributing towards RT physiological delay of Resp. output values)

            # 1) Calibration
            if all(~np.isnan(chest_raw[:cal])) and np.sum(np.diff(chest_raw[:cal])) != 0:  # Can;t be NaN or flatline
                UsingCal = True
                chest_raw_cal = chest_raw[:cal]
            else:
                UsingCal = False
                if np.sum(np.diff(chest_raw[:cal])) == 0:  # flatline
                    ind_start = np.argwhere(np.diff(chest_raw) != 0)[0, 0]
                    chest_raw_cal = chest_raw[ind_start:cal + ind_start]
                else:  # NaN
                    nan_inds = np.argwhere(np.isnan(chest_raw))[:, 0]
                    nan_inds_end = nan_inds[np.argwhere(np.diff(nan_inds) != 1)[:, 0]]
                    i = 1
                    if len(nan_inds_end) == 0:
                        chest_raw_cal = chest_raw[nan_inds[-1] + 1:nan_inds[-1] + 1 + cal]
                    elif len(nan_inds_end) == 1:
                        chest_raw_cal = chest_raw[nan_inds_end[i - 1] + 1:nan_inds_end[i - 1] + 1 + cal]
                    else:
                        while i < len(nan_inds_end):
                            if nan_inds_end[i] - nan_inds_end[i - 1] >= cal:
                                chest_raw_cal = chest_raw[nan_inds_end[i - 1] + 1:nan_inds_end[i - 1] + 1 + cal]
                                i = len(nan_inds_end)
                            else:
                                i = i + 1
                                chest_raw_cal = chest_raw[:cal]
                                UsingCal = True
                if not UsingCal:
                    print("\nCalibration section found, but not at start of record")

            # 1.A) Spectrum Analysis of Breathing Signal for Adaptive Filtering + Initialization
            f, Pxx = periodogram(chest_raw_cal - np.mean(chest_raw_cal), fs)
            Pxx_den = Pxx / max(Pxx)
            # #        # ----------------------------------------
            #         import matplotlib.pyplot as plt
            #         plt.figure(figsize=(15,7.5))
            #         plt.subplot(311)
            #         plt.plot(f,Pxx)
            #         plt.ylabel('Periodogram')
            #         plt.xlim(0,5)
            #         plt.grid(True)

            #         plt.subplot(312)
            #         plt.plot(f,Pxx_den)
            #         plt.ylabel('Periodogram Density (PD)')
            #         plt.xlim(0,5)
            #         plt.grid(True)

            #         plt.subplot(313)
            #         plt.plot(f[f<=1.5],Pxx[f<=1.5]/max(Pxx[f<=1.5]))
            #         plt.ylabel('PD 0-1.5Hz Band')
            #         plt.xlabel('Hz')
            #         plt.xlim(0,5)
            #         plt.grid(True)
            # #        # ----------------------------------------
            Pxx_den = Pxx[f <= 1.5] / max(Pxx[f <= 1.5])
            f = f[f <= 1.5]
            _temp = np.argwhere(Pxx_den >= 0.2)  # PSD within 0-1.5Hz band >= 5%
            _temp = np.vstack((_temp, np.argwhere(Pxx_den[0:np.argwhere(f >= 0.75)[0][0]] >= 0.1)))
            _temp = np.unique(np.sort(_temp, axis=0))
            if np.size(_temp) != 0:
                highcut = np.ceil(f[_temp[-1]] * 10) / 10  # np.round(f[_temp[-1][0]]+0.1,decimals=1)
                if highcut >= 1.0:
                    try:
                        i = 2
                        while f[_temp[-i]] > 1.0:
                            i = i + 1
                        ind_prior = _temp[-i]
                        ind_og = _temp[-(i - 1)]
                        if f[ind_og] - f[ind_prior] > 0.5:
                            ind_check = ind_og - 1
                            while Pxx_den[ind_check] < 0.05:
                                ind_check = ind_check - 1
                            highcut_use = np.ceil(f[ind_check - 1] * 10) / 10  # np.round(f[ind_check-1]+0.1,decimals=1)
                        else:
                            highcut_use = highcut
                    except:
                        highcut_use = highcut
                else:
                    highcut_use = highcut
                if highcut > 1.4:
                    highcut_use = 1.4
                elif highcut < 0.4:
                    highcut_use = 0.4
            else:
                # don't want to save this to prev highcut's since may be noise
                highcut_use = 1.0
            chest_filt_cal = Filter_breathing_signal(chest_raw_cal, fs, highcut_use)
            chest_highcut_cal = Filter_breathing_signal(chest_raw_cal, fs,
                                                        1.5)  # filtering with highcut set to 1.5Hz, which will be used for computing consistent breath amplitudes
            highcut_prev = highcut_use

            # 1.B) Breathing Peak/Valley Detection + Threshold Initialization
            if hw_v in (0, 1):
                prom_0 = 0.1
            else:
                prom_0 = 10  # !!!
            peaks, dic = find_peaks(chest_filt_cal, prominence=prom_0,
                                    distance=fs)  # e.g. 1*fs = min peak-to-peak distance must be greater than 1 sec i.e. 60 brpm

            if np.size(peaks) == 0:
                print(
                    "\nBreathing calibration error: No breaths detected")  # if no breaths were detected can't continue
                ora_br = np.empty((0, 3))
                inst_rVT_raw = np.empty((0, 5))
                inst_rVE_raw = np.empty((0, 3))
                win_rVE_raw = np.empty((0, 3))
                internal_load = np.nan
                internal_load_signal = np.empty((0, 2))
                valley_peak_ALL = np.empty((0, 2))
                in_ex_ratio = np.empty((0, 2))
                X_bbyb = []
            else:
                peaks_keep = []
                peak_Prom = []
                for k in range(0, len(peaks)):
                    if peaks[k] - fs < 0:
                        start = 0
                    else:
                        start = peaks[k] - fs
                    pos = np.diff(chest_filt_cal[start:peaks[k]])
                    if peaks[k] + fs > len(chest_filt_cal):
                        end = len(chest_filt_cal)
                    else:
                        end = peaks[k] + fs
                    neg = np.diff(chest_filt_cal[peaks[k]:end])
                    if (all(a > 0 for a in pos) and all(a < 0 for a in neg)):
                        peaks_keep.append(peaks[k])
                        peak_Prom.append(dic['prominences'][k])
                    elif (all(a > 0 for a in pos[-int(0.2 * fs):]) and all(a < 0 for a in neg[:int(
                            0.2 * fs)])):  # and all((chest_filt_cal[start],chest_filt_cal[end-1]) < chest_filt_cal[peaks[k]])):
                        ### Could improve this^ rule
                        if k == len(peaks) - 1:
                            if chest_filt_cal[peaks[k - 1] + (peaks[k] - peaks[k - 1]) // 2] < chest_filt_cal[peaks[k]]:
                                peaks_keep.append(peaks[k])
                                peak_Prom.append(dic['prominences'][k])
                        elif k == 0:
                            if chest_filt_cal[peaks[k] + (peaks[k + 1] - peaks[k]) // 2] < chest_filt_cal[peaks[k]]:
                                peaks_keep.append(peaks[k])
                                peak_Prom.append(dic['prominences'][k])
                        else:
                            if (chest_filt_cal[peaks[k - 1] + (peaks[k] - peaks[k - 1]) // 2] < chest_filt_cal[
                                peaks[k]]) and (
                                    chest_filt_cal[peaks[k] + (peaks[k + 1] - peaks[k]) // 2] < chest_filt_cal[
                                peaks[k]]):
                                peaks_keep.append(peaks[k])
                                peak_Prom.append(dic['prominences'][k])
                peaks_keep = np.array(peaks_keep, dtype=int)
                peak_Prom = np.array(peak_Prom, dtype=np.float64)
                _temp = np.argwhere(peaks_keep < cal - cut + int(
                    .250 * fs))  # remove peaks from distorted segment of filtered signal (i.e. last 'cut' duration of signal) + a 250ms leeway if last peak detected in the middle of a breath at the 's'-'cut' border (to later be mitigated in (C))
                peaks_keep = peaks_keep[_temp[:, 0]]
                valleys, _dic = find_peaks(-chest_filt_cal, distance=0.5 * fs)
                _temp = np.argwhere(
                    valleys < cal - cut + int(.250 * fs))  # similar as described above just for breath valleys
                valleys = valleys[_temp[:, 0]]
                valley_peak = []
                p_ind1 = 0
                v_ind0 = 0
                while p_ind1 < len(peaks_keep):
                    if peaks_keep[p_ind1] < valleys[v_ind0]:
                        p_ind1 = p_ind1 + 1
                    else:
                        if v_ind0 == len(valleys) - 1:
                            valley_peak.append([valleys[v_ind0], peaks_keep[p_ind1]])
                            p_ind1 = p_ind1 + 1
                            v_ind0 = v_ind0 + 1
                        elif peaks_keep[p_ind1] < valleys[v_ind0 + 1]:
                            valley_peak.append([valleys[v_ind0], peaks_keep[p_ind1]])
                            p_ind1 = p_ind1 + 1
                            v_ind0 = v_ind0 + 1
                        else:
                            v_ind0 = v_ind0 + 1
                valley_peak = np.array(valley_peak, dtype=int)  # index of peaks/valleys
                valley_peak_Range_highcut = chest_highcut_cal[
                    valley_peak]  # range values (i.e. digital signal value) of peaks/values
                valley_peak_ALL = valley_peak
                valley_peak_Range_highcut_ALL = valley_peak_Range_highcut
                if len(peaks_keep) > 1:
                    if len(peaks_keep) > 2:
                        indd = 1
                    else:
                        indd = 0
                    peak_Dis = np.diff(peaks_keep)  # [sample no.]
                    minDis = peak_Dis[
                        np.argsort(peak_Dis)[indd]]  # i.e. second smallest value [sample no.]   # min(peak_Dis)   #
                    _temp = np.argsort(peak_Prom)
                    minProm = peak_Prom[_temp[indd]]  # i.e. second smallest value    # min(peak_Prom)   #
                else:
                    minDis = 2 * fs / 0.75
                    minProm = prom_0
                v_prev = []
                v_prev_Range_highcut = []
                if valleys[-1] > valley_peak[-1, 1]:
                    v_prev = -int(cal - cut - valleys[-1])
                    v_prev_Range_highcut = chest_highcut_cal[valleys[-1]]
                # #            # ----------------------------------------
                #             import matplotlib.pyplot as plt

                #             plt.figure(figsize=(15,5))
                #             plt.subplot(211)
                #             plt.plot(chest_raw_cal)
                #             plt.legend(['highcut determined: %0.1f' %(highcut)],loc='upper right', fontsize=10)
                #             plt.grid(True)

                #             plt.subplot(212)
                #             plt.plot(chest_filt_cal)
                #             plt.plot(valley_peak_ALL[:,1], chest_filt_cal[valley_peak_ALL[:,1]],'ro')
                #             plt.plot(valley_peak_ALL[:,0], chest_filt_cal[valley_peak_ALL[:,0]],'go')
                #             plt.legend(['highcut used: %0.1f' %(highcut_use)],loc='upper right', fontsize=10)
                #             plt.grid(True)

                #             plt.show()
                # #            # ----------------------------------------

                # 2) Data
                if not UsingCal:
                    cal = w - s
                    valley_peak_ALL = np.empty((0, 2))
                    valley_peak_Range_highcut_ALL = np.empty((0, 2))
                    v_prev = []
                    v_prev_Range_highcut = []
                    minDis = 2 * fs / 0.75
                    peak_Dis = [minDis]
                    minProm = prom_0

                n = cal + s
                while n in range(cal + s, len(chest_raw) - s):  # int((17*60)*fs)):#
                    chest_raw_w = chest_raw[n - w:n]

                    # 2.A) Adaptive Bandpass Filtering of Breathing Signal
                    chest_2hz_w = Filter_breathing_signal(chest_raw_w, fs, 2, 0.4)  # 0.4-2.0 Hz
                    f, Pxx = periodogram(chest_2hz_w - np.mean(chest_2hz_w), fs)
                    Pxx_den = Pxx / max(Pxx)

                    # # ----------------------------------------
                    # import matplotlib.pyplot as plt

                    # plt.figure(figsize=(15,5))
                    # plt.subplot(211)
                    # plt.plot(f,Pxx)
                    # plt.ylabel('Periodogram')
                    # plt.legend((['0.75/2(minDis)=%.2fsec' % (0.75/2*minDis/fs)]))
                    # plt.xlim(0,2)
                    # plt.grid(True)

                    # plt.subplot(212)
                    # plt.plot(f,Pxx_den)
                    # plt.ylabel('Periodogram Density')
                    # plt.xlabel('Hz')
                    # plt.legend((['0.5(minProm)=%.2f' % (0.5*minProm)]))
                    # plt.xlim(0,2)
                    # plt.grid(True)
                    # # ----------------------------------------

                    _temp = np.argwhere(Pxx_den[0:np.argwhere(f > 1)[0][0]] >= 0.2)  # PSD within 0-1.5Hz band >= 5%
                    _temp = np.vstack((_temp, np.argwhere(f > 1)[0] + np.argwhere(
                        Pxx_den[np.argwhere(f > 1)[0][0]:np.argwhere(f >= 1.5)[0][0]] >= 0.5)))
                    _temp = np.unique(np.sort(_temp, axis=0))
                    if np.size(_temp) != 0:
                        highcut = np.ceil((f[_temp[-1]] + 0.05) * 10) / 10  # np.round(f[_temp[-1][0]]+0.1,decimals=1)
                        if highcut >= 1.0:
                            try:
                                ind_og = _temp[-1]
                                ind_prior = _temp[-2]
                                if f[ind_og] - f[ind_prior] > 0.5:
                                    ind_cut = ind_prior + 1
                                    p_band = np.sum(Pxx_den[ind_cut:ind_og - 1])
                                    while p_band > 0.2:
                                        ind_cut = ind_cut + 1
                                        p_band = np.sum(Pxx_den[ind_cut:ind_og - 1])
                                    highcut = np.ceil(
                                        (f[ind_cut] + 0.05) * 10) / 10  # np.round(f[ind_cut]+0.1,decimals=1)
                            except:
                                pass

                        if highcut > 1.5:
                            highcut = 1.5
                        #                    # Update prev highcut frequencies and select
                        if np.size(highcut_prev) >= int(np.round((s + cut) / s)):
                            highcut_prev = np.vstack((highcut, highcut_prev[:int(np.round(w / s) - 1)]))
                        else:
                            highcut_prev = np.vstack((highcut, highcut_prev))
                        highcut_use = max(highcut_prev)
                    #                    highcut_use = highcut
                    else:
                        # don't want to save this to prev highcut's since may be noise
                        highcut_use = 1.2
                    try:
                        chest_filt_w = Filter_breathing_signal(chest_raw_w, fs, highcut_use)
                    except:
                        chest_filt_w = Filter_breathing_signal(chest_raw_w, fs, highcut_use[0])
                    chest_highcut_w = Filter_breathing_signal(chest_raw_w, fs,
                                                              1.5)  # filtering with highcut set to 1.5Hz, which will be used for computing consistent breath amplitudes

                    # 2.B) Breathing Peak/Valley Detection with Adaptive Thresholds
                    peaks, dic = find_peaks(chest_filt_w, prominence=0.5 * minProm, distance=int(
                        0.75 / 2 * minDis))  # e.g. 1*fs = min peak-to-peak distance must be greater than 1 sec i.e. 60 brpm
                    valleys, _dic = find_peaks(-chest_filt_w, prominence=0.5 * minProm, distance=int(0.75 / 2 * minDis))

                    #                peak_Prom = np.hstack((peak_Prom,dic['prominences']))
                    #                if len(peak_Prom) > 11:
                    #                    peak_Prom = peak_Prom[-11:]
                    #                if min(peak_Prom) < prom_0:
                    #                    minProm = prom_0
                    #                else:
                    #                    minProm = min(peak_Prom)

                    # only keep peak/valleys in the new proceesing segment of the win + 250ms leeway
                    _temp_p = np.argwhere((peaks >= w - s - cut - int(.250 * fs)) & (peaks < w - cut + int(.250 * fs)))
                    _temp_v = np.argwhere(
                        (valleys >= w - s - cut - int(.250 * fs)) & (valleys < w - cut + int(.250 * fs)))
                    if np.size(_temp_p) > 0:

                        #                peak_Prom = np.hstack((peak_Prom,dic['prominences'][_temp_p][:,0]))
                        #                if len(peak_Prom) > 11: # minProm from last 11 breaths
                        #                    peak_Prom = peak_Prom[-11:]
                        #                minProm = min(peak_Prom)

                        if len(valley_peak_ALL) > 0:
                            if np.abs(n - w + peaks[_temp_p[0, 0]] - valley_peak_ALL[-1, 1]) < int(.250 * fs):
                                _temp_p = _temp_p[1:]
                        else:
                            _temp_p = np.argwhere((peaks >= int(.250 * fs)) & (peaks < w - cut + int(.250 * fs)))
                            _temp_v = np.argwhere((valleys >= int(.250 * fs)) & (valleys < w - cut + int(.250 * fs)))

                    # rules for accepting/rejecting peaks and valleys:
                    valley_peak_w = []
                    valley_peak_Range_highcut_w = []
                    p_inds_used = []
                    #                v_inds_used = []
                    if (np.size(_temp_p) > 0 and np.size(_temp_v) > 0):
                        peaks_add = peaks[_temp_p[:, 0]]
                        valleys_add = valleys[_temp_v[:, 0]]
                        p_ind1 = 0
                        v_ind0 = 0
                        if (peaks_add[p_ind1] < valleys_add[v_ind0] and np.size(v_prev) > 0):
                            if not n - w + peaks_add[p_ind1] < n - cut - s + v_prev:
                                valley_peak_w.append([n - cut - s + v_prev, n - w + peaks_add[p_ind1]])
                                p_inds_used.append(p_ind1)
                                valley_peak_Range_highcut_w.append(
                                    [v_prev_Range_highcut, chest_highcut_w[peaks_add[p_ind1]]])
                            p_ind1 = p_ind1 + 1
                        elif (peaks_add[p_ind1] < valleys_add[v_ind0] and np.size(v_prev) != 0):
                            p_ind1 = 1
                        while (p_ind1 < len(peaks_add)) and v_ind0 < len(valleys_add):
                            if peaks_add[p_ind1] < valleys_add[v_ind0]:
                                p_ind1 = p_ind1 + 1
                            else:
                                if v_ind0 == len(valleys_add) - 1:
                                    valley_peak_w.append([n - w + valleys_add[v_ind0], n - w + peaks_add[p_ind1]])
                                    p_inds_used.append(p_ind1)
                                    #                                v_inds_used.append(v_ind0)
                                    valley_peak_Range_highcut_w.append(
                                        [chest_highcut_w[valleys_add[v_ind0]], chest_highcut_w[peaks_add[p_ind1]]])
                                    p_ind1 = p_ind1 + 1
                                    v_ind0 = v_ind0 + 1
                                elif peaks_add[p_ind1] < valleys_add[v_ind0 + 1]:
                                    valley_peak_w.append([n - w + valleys_add[v_ind0], n - w + peaks_add[p_ind1]])
                                    p_inds_used.append(p_ind1)
                                    #                                v_inds_used.append(v_ind0)
                                    valley_peak_Range_highcut_w.append(
                                        [chest_highcut_w[valleys_add[v_ind0]], chest_highcut_w[peaks_add[p_ind1]]])
                                    p_ind1 = p_ind1 + 1
                                    v_ind0 = v_ind0 + 1
                                else:
                                    v_ind0 = v_ind0 + 1
                        valley_peak_w = np.array(valley_peak_w, dtype=int)
                        valley_peak_Range_highcut_w = np.array(valley_peak_Range_highcut_w, dtype=np.float64)
                        if np.size(valley_peak_w) != 0:
                            valley_peak_ALL = np.vstack((valley_peak_ALL, valley_peak_w))
                            valley_peak_Range_highcut_ALL = np.vstack(
                                (valley_peak_Range_highcut_ALL, valley_peak_Range_highcut_w))
                        v_prev = []
                        if valleys_add[-1] > peaks_add[-1]:
                            v_prev = int(valleys_add[-1] - (w - s))
                            v_prev_Range_highcut = chest_highcut_w[valleys_add[-1]]
                    elif (np.size(_temp_p) == 0 and np.size(_temp_v) > 0):
                        peaks_add = []
                        valleys_add = valleys[_temp_v[:, 0]]
                        v_prev = int(valleys_add[-1] - (w - s))
                        v_prev_Range_highcut = chest_highcut_w[valleys_add[-1]]
                    elif (np.size(_temp_p) > 0 and np.size(_temp_v) == 0):
                        peaks_add = peaks[_temp_p[:, 0]][0]
                        valleys_add = []
                        if np.size(v_prev) != 0:
                            valley_peak_w = np.array([n - cut - s + v_prev, n - w + peaks_add], dtype=int)
                            p_inds_used.append(0)
                            valley_peak_Range_highcut_w = np.array([v_prev_Range_highcut, chest_highcut_w[peaks_add]],
                                                                   dtype=np.float64)
                            valley_peak_ALL = np.vstack((valley_peak_ALL, valley_peak_w))
                            valley_peak_Range_highcut_ALL = np.vstack(
                                (valley_peak_Range_highcut_ALL, valley_peak_Range_highcut_w))
                        elif (all(a > (valley_peak_ALL[-1, 1] - (n - w)) for a in valleys) and all(a < w - cut for a in
                                                                                                   valleys)):  # search if there is a valley in b/w this peak and the prev peak which was missed during the RT processing
                            _temp_v2 = np.argwhere((valleys > valley_peak_ALL[-1, 1] - (n - w)) & (valleys < w - cut))
                            if np.size(_temp_v2) != 0:
                                missed_valley_search = valleys[_temp_v2[:, 0]][-1]
                                valley_peak_w = np.array([n - w + missed_valley_search, n - w + peaks_add], dtype=int)
                                p_inds_used.append(0)
                                valley_peak_Range_highcut_w = np.array(
                                    [chest_highcut_w[missed_valley_search], chest_highcut_w[peaks_add]],
                                    dtype=np.float64)
                                valley_peak_ALL = np.vstack((valley_peak_ALL, valley_peak_w))
                                valley_peak_Range_highcut_ALL = np.vstack(
                                    (valley_peak_Range_highcut_ALL, valley_peak_Range_highcut_w))
                            else:
                                peaks_add = []  # i.e. one peak after the other, so we don't consider this peak and wait for the next valley to then supercede a peak
                        else:
                            peaks_add = []  # i.e. one peak after the other, so we don't consider this peak and wait for the next valley to then supercede a peak
                        v_prev = []
                    elif (np.size(_temp_p) == 0 and np.size(_temp_v) == 0):
                        peaks_add = []
                        valleys_add = []
                        if np.size(v_prev) != 0:
                            v_prev = int(v_prev - s)

                    if not p_inds_used == []:
                        peak_Prom = np.hstack((peak_Prom, dic['prominences'][_temp_p[:, 0][p_inds_used]]))
                        if len(peak_Prom) > 11:
                            peak_Prom = peak_Prom[-11:]
                        if min(peak_Prom) < prom_0:
                            minProm = prom_0
                        else:
                            minProm = min(peak_Prom)

                    if len(valley_peak_ALL[:,
                           1]) > 10:  # minDis based on last 10 peak-to-peak, i.e. 11 breaths (could have also tried based on last 30sec for example)
                        peak_Dis = np.diff(valley_peak_ALL[-10:, 1])
                        minDis = min(peak_Dis)
                    elif len(valley_peak_ALL[:, 1]) >= 2:
                        peak_Dis = np.diff(valley_peak_ALL[:, 1])
                        minDis = min(peak_Dis)
                    if minDis < fs:
                        minDis = fs

                    # #            # ----------------------------------------
                    #             import matplotlib.pyplot as plt
                    #             time_w = time[n-w:n]

                    #             chest_raw_plot = chest_raw[:n]
                    #             chest_filt_plot = Filter_breathing_signal(chest_raw_plot,fs,highcut_use[0])

                    #             inc_xticks = 0.5
                    #             max_time = time_w[-1]
                    #             min_time = time_w[0]
                    #             if np.round(np.mod(np.ceil(max_time),inc_xticks)) != 0.0:
                    #                 xticks = np.linspace(np.floor(min_time),int(np.ceil(max_time)+inc_xticks-np.mod(np.ceil(max_time),inc_xticks)),int(np.ceil(max_time)/inc_xticks + (inc_xticks-np.mod(np.ceil(max_time),inc_xticks))/inc_xticks)+1)
                    #             else:
                    #                 xticks = np.linspace(np.floor(min_time),int(np.ceil(max_time)),int(np.ceil(max_time)/inc_xticks + 1))
                    #             xlabels=['%02i:%02i' % (np.floor(z),np.round((z-np.floor(z))*60)) for z in xticks]

                    #             fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15,6.6))
                    #             fig.subplots_adjust(hspace=0.15)
                    #             axs[0].plot(time_w*60,chest_raw_w, 'k')
                    #             axs[0].grid(True)
                    #             axs[0].set_ylabel('Raw', fontsize=12)
                    #             axs[0].legend(['highcut determined: %0.1f Hz' %(highcut)], fontsize=10, loc='upper left')
                    #             axs[1].plot(time_w*60,chest_filt_w,'b-',linewidth=1.5,alpha=0.5)
                    #             axs[1].grid(True)
                    #             axs[1].plot(time_w[w-s-cut:w-cut]*60,chest_filt_w[w-s-cut:w-cut],'y-',linewidth=5,alpha=0.3)
                    #             axs[1].plot(time_w[peaks]*60,chest_filt_w[peaks],'k*',markeredgecolor='k', markersize=10)
                    #             axs[1].plot(time_w[valleys]*60,chest_filt_w[valleys],'k.', markersize=14)
                    #             axs[1].plot(time_w[peaks_add]*60,chest_filt_w[peaks_add],'r*',markeredgecolor='r', markersize=10)
                    #             axs[1].plot(time_w[valleys_add]*60,chest_filt_w[valleys_add],'g.', markersize=14)
                    # #            axs[1].plot(time_w[valley_peak_w[:,1]],chest_highcut_w[valley_peak_w[:,1]],'r*',markeredgecolor='r', markersize=10)
                    # #            axs[1].plot(time_w[valley_peak_w[:,0]],chest_highcut_w[valley_peak_w[:,0]],'g.', markersize=14)
                    #             axs[1].set_ylabel('Filtered', fontsize=12)
                    #             axs[1].legend(['highcut used: %0.1f Hz' % (highcut_use)], fontsize=10, loc='upper left')
                    #             axs[1].set_xlabel('Time [$sec$]', fontsize=12)
                    #             axs[1].set_xlim(time_w[0]*60,time_w[-1]*60)
                    #             plt.show()

                    # #            plt.figure(figsize=(15,5))
                    # #            plt.subplot(211)
                    # #            plt.plot(chest_raw_plot)
                    # #            plt.grid(True)
                    # #            plt.xlim(n-60*fs,n)
                    # #
                    # #            plt.subplot(212)
                    # #            plt.plot(chest_filt_plot)
                    # #            plt.hold(True)
                    # #            plt.plot(valley_peak_ALL[:,1], chest_filt_plot[valley_peak_ALL[:,1]],'ro')
                    # #            plt.plot(valley_peak_ALL[:,0], chest_filt_plot[valley_peak_ALL[:,0]],'go')
                    # #            plt.grid(True)
                    # #            plt.xlim(n-60*fs,n)
                    # #
                    # #            plt.show()
                    # #
                    #             print(highcut_prev)
                    # #            print(highcut_use)
                    # #        # ----------------------------------------

                    # Prep for next window
                    n = n + s

                # DISCARD FALSE POSITIVE BREATHS DETECTED
                # A and B) Breaths who's peak detected is lower than the proceeding AND subsequent valleys detected
                min_amp = 25
                #            vpv_highcut = valley_peak_Range_highcut_ALL[:-1,:]
                #            vpv_highcut = np.column_stack((vpv_highcut,valley_peak_Range_highcut_ALL[1:,0]))
                #            ab = np.argwhere((vpv_highcut[:,1]>vpv_highcut[:,0]+min_amp) & (vpv_highcut[:,1]>vpv_highcut[:,2]+min_amp))[:,0]
                #            valley_peak_Range_highcut_ALL = np.vstack((valley_peak_Range_highcut_ALL[ab,:],valley_peak_Range_highcut_ALL[-1,:]))
                #            valley_peak_ALL = np.vstack((valley_peak_ALL[ab,:],valley_peak_ALL[-1,:]))
                # A) Breaths who's peak detected is lower than proceeding valley detected
                a = np.argwhere(valley_peak_Range_highcut_ALL[:, 1] > valley_peak_Range_highcut_ALL[:, 0] + min_amp)[:,
                    0]  # only keep breaths detected where the peak detected is larger than the proceeding valley detected
                valley_peak_Range_highcut_ALL = valley_peak_Range_highcut_ALL[a, :]
                valley_peak_ALL = valley_peak_ALL[a, :]
                #            # B) Breaths who's peak detected is lower than subsequent valley detected
                #            b = np.argwhere(valley_peak_Range_highcut_ALL[1:,1]>valley_peak_Range_highcut_ALL[:-1,0] + min_amp)[:,0] # only keep breaths detected where the peak detectedis larger than the proceeding valley detected
                #            valley_peak_Range_highcut_ALL = valley_peak_Range_highcut_ALL[b,:]
                #            valley_peak_ALL = valley_peak_ALL[b,:]
                #            vp_final = [valley_peak_ALL[0,:],valley_peak_Range_highcut_ALL[0,:]]
                #            vp_final = [[valley_peak_ALL[0,0],valley_peak_ALL[0,1]],[valley_peak_Range_highcut_ALL[0,0],valley_peak_Range_highcut_ALL[0,1]]]
                # Remove breaths with amp less than min_amp:
                vp_final = []
                vp_final.append([valley_peak_ALL[0, 0], valley_peak_ALL[0, 1], valley_peak_Range_highcut_ALL[0, 0],
                                 valley_peak_Range_highcut_ALL[0, 1]])
                i = 1
                while i < len(valley_peak_ALL) - 1:
                    if valley_peak_Range_highcut_ALL[i, 1] < valley_peak_Range_highcut_ALL[i + 1, 0] + min_amp:
                        vp_final.append([valley_peak_ALL[i, 0],
                                         valley_peak_ALL[i + 1, 1],
                                         valley_peak_Range_highcut_ALL[i, 0],
                                         valley_peak_Range_highcut_ALL[i + 1, 1]])
                        i = i + 2
                    else:
                        vp_final.append(
                            [valley_peak_ALL[i, 0], valley_peak_ALL[i, 1], valley_peak_Range_highcut_ALL[i, 0],
                             valley_peak_Range_highcut_ALL[i, 1]])
                        i = i + 1
                vp_final = np.array(vp_final, dtype=np.float)
                # Remove upper-end outlier breath volumes:
                temp = np.diff(vp_final[:, 2:])[:, 0]
                mean_ALL = np.mean(temp)
                std_ALL = np.std(temp)
                temp2 = np.abs(temp - mean_ALL)
                '''
                np.argwhere( temp > 4 * mean_ALL )[:,0]
                np.argwhere( temp2 > 4 * std_ALL )[:,0]
                '''
                factor = 4
                # vp_final = vp_final[temp<=factor*mean_ALL,:]
                vp_final = vp_final[temp2 <= factor * std_ALL, :]
                valley_peak_ALL = np.array(vp_final[:, 0:2], dtype=int)
                valley_peak_Range_highcut_ALL = vp_final[:, 2:]

                # C) Significant breath-to-breath outliers

                if np.size(valley_peak_ALL[:, 1]) < 2:
                    ora_br = np.empty((0, 3))
                    inst_rVT_raw = np.empty((0, 5))
                    inst_rVE_raw = np.empty((0, 3))
                    win_rVE_raw = np.empty((0, 3))
                    internal_load = np.nan
                    internal_load_signal = np.empty((0, 2))
                    in_ex_ratio = np.empty((0, 2))
                    X_bbyb = []
                else:
                    #            pk_temp = valley_peak_ALL[:,1]
                    #            br_temp = fs * 60 / np.diff(pk_temp)
                    #            cont = True
                    #            while cont:
                    #                keep = []
                    #                disc = []
                    #                for r in range(1,len(br_temp)):
                    #                    if (br_temp[r] - br_temp[r-1]) >= 20:
                    #                        disc.append(r)
                    #                    elif (br_temp[r] - br_temp[r-1]) < br_temp[r-1]:
                    #                        keep.append(r)
                    #                    else:
                    #                        disc.append(r)
                    #                keep = np.array(keep,dtype=int)
                    #                pk_temp = pk_temp[np.hstack(([0,1],keep+1))]
                    #                br_temp = fs * 60 / np.diff(pk_temp)
                    #                if len(disc)== 0:
                    #                    cont = False
                    #            keep2 = np.argwhere(br_temp >= 8)
                    #            inst_br_MOD = br_temp[keep2][:,0]
                    #            inst_br_time_MOD = pk_temp[1:] / 60 / fs
                    #            inst_br_time_MOD = inst_br_time_MOD[keep2][:,0]

                    # PEAK=TO-PEAK
                    # 3) Breath-by-breath Inst + Outlier-rejection Avg BR Calculations
                    inst_br = fs * 60 / np.diff(valley_peak_ALL[:, 1])
                    #                    inst_br[inst_br<2] = np.nan
                    inst_br_ind = valley_peak_ALL[1:, 1]
                    inst_br_time = inst_br_ind / 60 / fs

                    #                    try:
                    #                        nan_time = inst_br_time[np.isnan(inst_br)][0]
                    #                    except:
                    #                        pass

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
                    #                    try:
                    #                        ind_nan = np.argwhere(np.round(br_time,4)==np.round(nan_time,4))[0][0]
                    #                        br[ind_nan:ind_nan+br_win] = np.nan
                    #                    except:
                    #                        pass

                    # 4) Breath-by-breath Inst + Outlier-rejection Avg rVE -&- 15sec window rVE Calculations

                    # ---------------------------------------------------------------
                    # Extracting Breath Amplitude (delta_amp) from Raw Breathing Signal instead of Filtered
                    # Searching X seconds before and after peak, where X is dependant on BR
                    # Skipping the step of defining variable valley_peak_RANGE_raw_ALL
                    delta_amp_raw_og = []
                    raw_valley = []
                    raw_peak = []

                    #                    delta_amp_inex = [] # !!! amp of inhale + exhale amplitudes

                    #                import matplotlib.pyplot as plt
                    #                time = np.array([t/fs/60 for t in range(len(chest_raw))]) # [min]
                    #                plt.plot(time,chest_raw,'k-',lw=0.75)
                    #                plt.show()

                    for i in range(1, len(inst_br) + 1):
                        x = 60 / inst_br[
                            i - 1]  # breath-to-breath duration [sec] of the i-th breath (i.e. i-th minus (i-1)-th peak-to-peak rate)
                        if ~np.isnan(x):
                            x = int(np.ceil((
                                                        1 / 12) * x * fs))  # search +/- 1/12 of the period length (i.e. pie/6), then convert to [sample no.]
                            raw_value_valley = np.mean(chest_raw[np.arange(valley_peak_ALL[i, 0] - x,
                                                                           valley_peak_ALL[i, 0] + x, 1,
                                                                           dtype=int)])  # chest_raw[valley_peak_ALL[i,0]] #
                            raw_value_peak = np.mean(chest_raw[
                                                         np.arange(valley_peak_ALL[i, 1] - x, valley_peak_ALL[i, 1] + x,
                                                                   1)])  # chest_raw[valley_peak_ALL[i,1]] #
                            delta_amp_raw_og.append(raw_value_peak - raw_value_valley)
                            raw_valley.append(raw_value_valley)
                            raw_peak.append(raw_value_peak)
                        #                            try:
                        #                                raw_value_valley_next = np.mean(chest_raw[np.arange(valley_peak_ALL[i+1,0]-x,valley_peak_ALL[i+1,0]+x,1,dtype=int)]) # chest_raw[valley_peak_ALL[i+1,0]] #
                        #                                delta_amp_inex.append(raw_value_peak-raw_value_valley + raw_value_peak-raw_value_valley_next)
                        #                            except:
                        #                                pass
                        else:
                            delta_amp_raw_og.append(np.nan)
                            raw_valley.append(np.nan)
                            raw_peak.append(np.nan)
                    #                            delta_amp_inex.append(np.nan)
                    #                    delta_amp_raw_og = np.array(np.sqrt(delta_amp_raw_og),dtype=np.float)
                    delta_amp_raw_og = np.array(delta_amp_raw_og, dtype=np.float)
                    raw_valley = np.array(raw_valley, dtype=np.float)
                    raw_peak = np.array(raw_peak, dtype=np.float)
                    keep_og = np.argwhere((delta_amp_raw_og >= 0) | (np.isnan(delta_amp_raw_og)))[:, 0]
                    #                    keep_og = np.argwhere((delta_amp_raw_og>=0))[:,0]
                    delta_amp_raw = delta_amp_raw_og[keep_og]
                    inst_br_temp = inst_br[keep_og]
                    inst_br_ind_temp = inst_br_ind[keep_og]
                    inst_rVE_raw = []
                    for k in range(0, len(inst_br_temp)):
                        inst_rVE_raw.append(inst_br_temp[k] * delta_amp_raw[k] / 100)
                    inst_rVE_raw = np.array(inst_rVE_raw, dtype=np.float)
                    inst_rVE_time_raw = inst_br_ind_temp / fs / 60  # [min]

                    #                    delta_amp_inex = np.array(delta_amp_inex,dtype=np.float)

                    #                    fig,ax = plt.subplots(figsize=(18,6))
                    ##                    ax.plot(inst_rVE_time_raw,delta_amp_raw,'b.',alpha=0.2)
                    #                    ax.plot(inst_rVE_time_raw,delta_amp_raw_smooth,'b-',alpha=0.5,lw=1.5,label='Just amp of valley to peak')
                    #                    ax.legend(loc='upper left',fontsize=10)
                    #                    ax2 = ax.twinx()
                    ##                    ax2.plot(inst_br_time[:-1],delta_amp_inex,'g.',alpha=0.2)
                    #                    ax2.plot(inst_br_time[:-1],delta_amp_inex_smooth,'g-',alpha=0.5,lw=1.5,label='Amp of valley to peak + peak to valley')
                    #                    ax2.legend(loc='upper right',fontsize=10)
                    #                    ax.grid(True)
                    #                    ax.set_xlim(0,inst_br_time[-1])

                    ''' Insert NaN where there were 'pause-resume'(s): '''
                    make_nan = np.argwhere(np.isnan(delta_amp_raw_og))[:, 0]
                    if make_nan.size != 0:  # if not empty
                        inst_br[make_nan] = np.nan
                        br[make_nan - br_win + 1] = np.nan

                    # ---------------------------------------------------------------

                    delta_amp = []
                    #        delta_amp_raw2 = []
                    for k in range(1, len(inst_br) + 1):
                        delta_amp.append(valley_peak_Range_highcut_ALL[k, 1] - valley_peak_Range_highcut_ALL[k, 0])
                    #            delta_amp_raw2.append(chest_raw[valley_peak_ALL[k,1]] - chest_raw[valley_peak_ALL[k,0]])
                    delta_amp = np.array(np.sqrt(delta_amp), dtype=np.float)
                    #        delta_amp_raw2 = np.array(np.sqrt(delta_amp_raw2),dtype=np.float)
                    inst_rVE = []
                    for k in range(0, len(inst_br)):
                        inst_rVE.append(inst_br[k] * delta_amp[k] / 100)
                    inst_rVE = np.array(inst_rVE, dtype=np.float)
                    # inst_rVE_time = inst_br_ind / fs / 60 # [min]

                    #        keep = np.argwhere((delta_amp_raw2>=0))[:,0]
                    #        delta_amp_raw2 = delta_amp_raw2[keep]
                    #        inst_br_temp = inst_br[keep]
                    #        inst_br_ind_temp = inst_br_ind[keep]
                    #        inst_rVE_raw2 = []
                    #        for k in range(0,len(inst_br_temp)):
                    #            inst_rVE_raw2.append(inst_br_temp[k] * delta_amp_raw2[k]/100)
                    #        inst_rVE_raw2 = np.array(inst_rVE_raw2,dtype=np.float)
                    #        inst_rVE_time_raw2 = inst_br_ind_temp / fs / 60 # [min]

                    #            ora_rVE = []
                    #            k = 0
                    #            while k <= len(inst_br)-br_win:
                    #                x = inst_rVE[k:k+br_win]
                    #                mean_x = np.mean(x)
                    #                y = np.abs(x - mean_x)
                    #                a = np.argsort(y)
                    #                keep = a[:-br_reject]
                    #                avg_rVE = np.mean(x[keep])
                    #                ora_rVE.append(avg_rVE)
                    #                k = k + 1
                    #            ora_rVE = np.array(ora_rVE,dtype=np.float) # [brpm]
                    #            ora_rVE_time = valley_peak_ALL[br_win:,1] / fs / 60 # [min]

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
                    #                    try:
                    #                        nan_start, nan_end = int(np.floor(br_time[ind_nan-1]*60 - win/fs)), int(np.ceil(br_time[ind_nan]*60 - win/fs))
                    #                        rVE[nan_start:nan_end] = np.nan
                    #                    except:
                    #                        pass
                    # rVE_time = np.linspace(win/fs/60, ((k+win)/fs-1)/60, int(k/fs)) # [min]

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
                    #                    try:
                    #                        nan_start, nan_end = int(np.floor(br_time[ind_nan-1]*60 - win/fs)), int(np.ceil(br_time[ind_nan]*60 - win/fs))
                    #                        rVE_raw[nan_start:nan_end] = np.nan
                    #                    except:
                    #                        pass
                    rVE_raw_time = np.linspace(win / fs / 60, ((k + win) / fs - 1) / 60, int(k / fs))  # [min]

                    # Smoothing
                    if len(br) >= 31:
                        br_smooth = savgol_filter(br, 31, 1)
                    else:
                        br_smooth = br
                    #            if len(rVE) >= 61:
                    #                rVE_smooth = savgol_filter(rVE, 61, 1)
                    #            else:
                    #                rVE_smooth = rVE
                    if len(rVE_raw) >= 61:
                        rVE_raw_smooth = savgol_filter(rVE_raw, 61, 1)
                    else:
                        rVE_raw_smooth = rVE_raw
                    #            if len(inst_rVE) >= 45:
                    #                inst_rVE_smooth = savgol_filter(inst_rVE, 45, 1)
                    #                delta_amp_smooth = savgol_filter(delta_amp, 45, 1)
                    #            else:
                    #                inst_rVE_smooth = inst_rVE
                    #                delta_amp_smooth = delta_amp
                    if len(inst_rVE_raw) >= 21:
                        inst_rVE_smooth_raw = savgol_filter(inst_rVE_raw, 21, 1)
                        delta_amp_raw_smooth = savgol_filter(delta_amp_raw, 21, 1)
                    #                        delta_amp_inex_smooth = savgol_filter(delta_amp_inex, 21, 1)
                    else:
                        inst_rVE_smooth_raw = inst_rVE_raw
                        delta_amp_raw_smooth = delta_amp_raw
                    #        if len(inst_rVE_raw2) >= 31:
                    #            inst_rVE_smooth_raw2 = savgol_filter(inst_rVE_raw2, 31, 1)
                    #            delta_amp_raw2_smooth = savgol_filter(delta_amp_raw2, 31, 1)
                    #        else:
                    #            inst_rVE_smooth_raw2 = inst_rVE_raw2
                    #            delta_amp_raw2_smooth = delta_amp_raw2
                    #            if len(ora_rVE) >= 17:
                    #                ora_rVE_smooth = savgol_filter(ora_rVE, 17, 1)
                    #            else:
                    #                ora_rVE_smooth = ora_rVE

                    #                    # ----------------------------------------
                    #                    import matplotlib.pyplot as plt
                    #
                    #                    chest_filt = Filter_breathing_signal(chest_raw,fs,1.2)
                    #
                    #                    fig, axs = plt.subplots(3, 1, figsize=(12,9), sharex=True)
                    #                    fig.subplots_adjust(hspace=0.2)
                    #                    i = 0
                    #                    axs[i].plot(time,chest_filt,'b-',linewidth=1,alpha=0.75)
                    #                    axs[i].hold(True)
                    #                    axs[i].plot(time[valley_peak_ALL[1:,0]],chest_filt[valley_peak_ALL[1:,0]],'r.',alpha=0.75)
                    #                    axs[i].plot(time[valley_peak_ALL[1:,1]],chest_filt[valley_peak_ALL[1:,1]],'g.',alpha=0.75)
                    #                    axs[i].legend(('signal','filt valley','filt peak'), loc='upper left',fontsize=10)
                    #                    axs[i].set_ylabel('Filterred Breathing Signal',fontsize=12)
                    #                    axs[i].grid(True)
                    #                    axs[i].set_xlim(0,time[-1])
                    #                    i = i + 1
                    #                    axs[i].plot(time,chest_raw,'k-',linewidth=1,alpha=0.75)
                    #                    axs[i].hold(True)
                    #                    axs[i].plot(time[valley_peak_ALL[1:,0]],raw_valley,'o',mec='r',mfc='None',mew=1.2,alpha=0.75)
                    #                    axs[i].plot(time[valley_peak_ALL[1:,1]],raw_peak,'o',mec='g',mfc='None',mew=1.2,alpha=0.75)
                    #                    axs[i].plot(time[valley_peak_ALL[1:,0]],chest_raw[valley_peak_ALL[1:,0]],'r.',alpha=0.75)
                    #                    axs[i].plot(time[valley_peak_ALL[1:,1]],chest_raw[valley_peak_ALL[1:,1]],'g.',alpha=0.75)
                    #                    axs[i].legend(('signal','raw valley (avg.)','raw peak (avg.)','raw 2 valley','raw 2 peak'), loc='upper left',fontsize=10)
                    #                    axs[i].set_ylabel('Raw Breathing Signal',fontsize=12)
                    #                    axs[i].grid(True)
                    #                    axs[i].set_xlim(0,time[-1])
                    #                    i = i + 1
                    #                    axs[i].plot(inst_rVE_time_raw,delta_amp_raw,'c.',alpha=0.1)
                    #                    axs[i].hold(True)
                    #                    axs[i].plot(inst_rVE_time_raw,delta_amp_raw_smooth,'c-',linewidth=1.5,alpha=0.75)
                    #                    axs[i].plot(inst_rVE_time_raw2,delta_amp_raw2,'g.',alpha=0.1)
                    #                    axs[i].plot(inst_rVE_time_raw2,delta_amp_raw2_smooth,'g-',linewidth=1.5,alpha=0.75)
                    #                    axs[i].plot(inst_rVE_time,delta_amp,'m.',alpha=0.1)
                    #                    axs[i].plot(inst_rVE_time,delta_amp_smooth,'m-',linewidth=1.5,alpha=0.75)
                    #                    axs[i].legend(('raw inst','raw smooth','raw 2 inst','raw 2 smooth','filt inst','filt smooth'), loc='upper left',fontsize=10)
                    #                    axs[i].set_ylabel('VT ($sqrt(delta amp)$)',fontsize=12)
                    #                    axs[i].grid(True)
                    #                    axs[i].set_xlim(0,time[-1])
                    #                    axs[i].set_xlabel('Time [min]',fontsize=12)
                    #                    plt.show()
                    #
                    #        inc_xticks = 2
                    #        max_time = time[-1]
                    #        if np.round(np.mod(np.ceil(max_time),inc_xticks)) != 0.0:
                    #            xticks = np.linspace(0,int(np.ceil(max_time)+inc_xticks-np.mod(np.ceil(max_time),inc_xticks)),int(np.ceil(max_time)/inc_xticks + (inc_xticks-np.mod(np.ceil(max_time),inc_xticks))/inc_xticks)+1)
                    #        else:
                    #            xticks = np.linspace(0,int(np.ceil(max_time)),int(np.ceil(max_time)/inc_xticks + 1))
                    #        xlabels=['%02i:%02i' % (np.floor(z),np.round((z-np.floor(z))*60)) for z in xticks]
                    #
                    #
                    #
                    #        plt.subplots(figsize=(12,8))
                    #        plt.subplots_adjust(hspace=0.30)
                    ##        plt.suptitle('-- Filename: %s --' % (json_file), fontsize=14, y=0.97)
                    #
                    #        ax1 = plt.subplot(211)
                    #        ax1.plot(time,chest_raw,'k-')
                    #        ax1.set_title('Raw Breathing Signal',fontsize=13)
                    #        ax1.set_ylabel('Raw Signal',fontsize=12)
                    #        ax1.set_xticks(xticks)
                    #        ax1.set_xticklabels(xlabels,rotation='vertical')
                    #        ax1.grid(True)
                    #
                    #        ax2 = plt.subplot(212, sharex = ax1)
                    #        ax2.plot(time,chest_filt,'b-')
                    #        ax2.hold(True)
                    #        ax2.plot(time[valley_peak_ALL[:,1]],chest_filt[valley_peak_ALL[:,1]],'r.')
                    #        ax2.plot(time[valley_peak_ALL[:,0]],chest_filt[valley_peak_ALL[:,0]],'g.')
                    #        ax2.set_title('Filtered Breathing Signal',fontsize=13)
                    #        ax2.legend(('0.1-1.2Hz filtered signal','peaks','valleys'), loc='upper left', fontsize=10)
                    #        ax2.set_ylabel('Filtered Signal',fontsize=12)
                    #        ax2.set_xlabel('Time [mm:ss]',fontsize=12)
                    #        ax2.set_xticks(xticks)
                    #        ax2.set_xticklabels(xlabels,rotation='vertical')
                    #        ax2.grid(True)
                    #
                    #        #plt.show()
                    #
                    #        plt.subplots(figsize=(12,10),sharex=True)
                    #        plt.subplots_adjust(hspace=0.35)
                    #        plt.suptitle('-- Filename: %s --' % (json_file), fontsize=14, y=0.95)
                    #
                    #        plt.subplot(311)
                    #        plt.plot(inst_br_time, inst_br, 'm.', alpha=0.3)
                    #        plt.hold(True)
                    #        plt.plot(br_time, br, 'm-', linewidth=1)
                    #        plt.title('breath-by-breath Breathing Rate -- PEAK-TO-PEAK',fontsize=13)
                    #        plt.legend(('Inst.','%i/%i Avg' %(br_win-br_reject, br_win)), loc='upper left', fontsize=10)
                    #        plt.ylabel('BR [brpm]',fontsize=12)
                    #        plt.xticks(xticks,xlabels,rotation='vertical')
                    #        plt.grid(True)
                    #        plt.xlim(0,xticks[-1])
                    #
                    #        plt.subplot(312)
                    #        plt.plot(ora_rVE_time, ora_rVE, 'g.', alpha=0.3)
                    #        plt.hold(True)
                    ##        plt.plot(inst_rVE_time, inst_rVE_smooth, 'g-', linewidth=1.5, alpha=0.9)
                    #        plt.plot(ora_rVE_time, ora_rVE_smooth, 'g-', linewidth=1.5)
                    #        plt.title('breath-by-breath Minute Volume -- PEAK-TO-PEAK',fontsize=13)
                    #        plt.legend(('%i/%i Avg' %(br_win-br_reject, br_win),'Smoothed %i/%i Avg' %(br_win-br_reject, br_win)), loc='upper left', fontsize=10)
                    #        plt.ylabel('rVE',fontsize=12)
                    #        plt.xticks(xticks,xlabels,rotation='vertical')
                    #        plt.grid(True)
                    #        plt.xlim(0,xticks[-1])
                    #
                    #        plt.subplot(313)
                    #        plt.plot(rVE_time, rVE, 'b.', alpha=0.3)
                    #        plt.plot(rVE_time, rVE_smooth, 'b-', linewidth=1.5)
                    #        plt.title('sec-by-sec Minute Volume (%isec-window) -- PEAK-TO-PEAK' % (win/fs),fontsize=13)
                    #        plt.legend(('Inst.','Smoothed'), loc='upper left', fontsize=10)
                    #        plt.ylabel('rVE',fontsize=12)
                    #        plt.xlabel('Time [mm:ss]',fontsize=12)
                    #        plt.xticks(xticks,xlabels,rotation='vertical')
                    #        plt.grid(True)
                    #        plt.xlim(0,xticks[-1])
                    #
                    #        plt.show()
                    # ----------------------------------------

                    # VALLEY=TO-VALLEY
                    #            # 3) Breath-by-breath Inst + Outlier-rejection Avg BR Calculations
                    #            inst_br2 = fs * 60 / np.diff(valley_peak_ALL[:,0])
                    #            inst_br2_ind = valley_peak_ALL[1:,0]
                    #            inst_br2_time = inst_br2_ind / 60 / fs
                    #
                    #    #        br_win = 5
                    #    #        br_reject = 2
                    #            br2 = []
                    #            k = 0
                    #            while k <= len(inst_br2)-br_win:
                    #                x = inst_br2[k:k+br_win]
                    #                mean_x = np.mean(x)
                    #                y = np.abs(x - mean_x)
                    #                a = np.argsort(y)
                    #                keep = a[:-br_reject]
                    #                avg_br = np.mean(x[keep])
                    #                br2.append(avg_br)
                    #                k = k + 1
                    #            br2 = np.array(br2,dtype=int) # [brpm]
                    #            br2_time = valley_peak_ALL[br_win:,0] / fs / 60 # [min]
                    #
                    #            # 4) Breath-by-breath Inst + Outlier-rejection Avg rVE -&- 15sec window rVE Calculations
                    #            delta_amp = []
                    #            for k in range(0,len(inst_br2)):
                    #    #            delta_amp.append(np.mean(valley_peak_Range_highcut_ALL[k:k+1,1]) - valley_peak_Range_highcut_ALL[k,0])
                    #                delta_amp.append(valley_peak_Range_highcut_ALL[k,1] - valley_peak_Range_highcut_ALL[k,0])
                    #            delta_amp = np.array(np.sqrt(delta_amp),dtype=np.float)
                    #            inst_rVE2 = []
                    #            for k in range(0,len(inst_br2)):
                    #                inst_rVE2.append(inst_br2[k] * delta_amp[k]/100)
                    #            inst_rVE2 = np.array(inst_rVE2,dtype=np.float)
                    #            inst_rVE2_time = inst_br2_ind / fs / 60 # [min]
                    #
                    #            ora_rVE2 = []
                    #            k = 0
                    #            while k <= len(inst_br2)-br_win:
                    #                x = inst_rVE2[k:k+br_win]
                    #                mean_x = np.mean(x)
                    #                y = np.abs(x - mean_x)
                    #                a = np.argsort(y)
                    #                keep = a[:-br_reject]
                    #                avg_rVE = np.mean(x[keep])
                    #                ora_rVE2.append(avg_rVE)
                    #                k = k + 1
                    #            ora_rVE2 = np.array(ora_rVE2,dtype=np.float) # [brpm]
                    #            ora_rVE2_time = valley_peak_ALL[br_win:,1] / fs / 60 # [min]
                    #
                    #            win = 15*fs
                    #            slide = 1*fs
                    #            rVE2 = []
                    #            k = 0
                    #            while k < inst_br2_ind[-1] - np.mod(inst_br2_ind[-1],fs) - win + slide:
                    #                keep = np.argwhere((inst_br2_ind >= k) & (inst_br2_ind < k+win))
                    #                rVE2.append(np.sum(inst_rVE2[keep])/10 * 60/(win/fs))
                    #                k = k + slide
                    #            rVE2 = np.array(rVE2,dtype=np.float)
                    #            rVE2_time = np.linspace(win/fs/60, ((k+win)/fs-1)/60, int(k/fs)) # [min]
                    #
                    #            # Smoothing
                    #            if len(rVE2) >= 61:
                    #                rVE2_smooth = savgol_filter(rVE2, 61, 1)
                    #            else:
                    #                rVE2_smooth = rVE2
                    #            if len(inst_rVE2) >= 21:
                    #                inst_rVE2_smooth = savgol_filter(inst_rVE2, 21, 1)
                    #            else:
                    #                inst_rVE2_smooth = inst_rVE2
                    #            if len(ora_rVE2) >= 17:
                    #                ora_rVE2_smooth = savgol_filter(ora_rVE2, 17, 1)
                    #            else:
                    #                ora_rVE2_smooth = ora_rVE2
                    #
                    #            # ----------------------------------------
                    #    #        import matplotlib.pyplot as plt
                    #    #
                    #    #        inc_xticks = 0.5
                    #    #        max_time = time[-1]
                    #    #        if np.round(np.mod(np.ceil(max_time),inc_xticks)) != 0.0:
                    #    #            xticks = np.linspace(0,int(np.ceil(max_time)+inc_xticks-np.mod(np.ceil(max_time),inc_xticks)),int(np.ceil(max_time)/inc_xticks + (inc_xticks-np.mod(np.ceil(max_time),inc_xticks))/inc_xticks)+1)
                    #    #        else:
                    #    #            xticks = np.linspace(0,int(np.ceil(max_time)),int(np.ceil(max_time)/inc_xticks + 1))
                    #    #        xlabels=['%02i:%02i' % (np.floor(z),np.round((z-np.floor(z))*60)) for z in xticks]
                    #    #
                    #    #        plt.subplots(figsize=(12,12),sharex=True)
                    #    #        plt.subplots_adjust(hspace=0.35)
                    #    #        #plt.subplots(sharex=True)
                    #    #        plt.suptitle('-- Filename: %s --' % (json_file), fontsize=14, y=0.95)
                    #    #
                    #    #        plt.subplot(311)
                    #    #        plt.plot(inst_br2_time, inst_br2, 'm.', alpha=0.3)
                    #    #        plt.hold(True)
                    #    #        plt.plot(br2_time, br2, 'm-', linewidth=1)
                    #    #        plt.title('breath-by-breath Breathing Rate -- VALLEY-TO-VALLEY',fontsize=13)
                    #    #        plt.legend(('Inst.','%i/%i Avg' %(br_win-br_reject, br_win)), loc='upper left', fontsize=10)
                    #    #        plt.ylabel('BR [brpm]',fontsize=12)
                    #    #        plt.xticks(xticks,xlabels,rotation='vertical')
                    #    #        plt.grid(True)
                    #    #        plt.xlim(0,xticks[-1])
                    #    #
                    #    #        plt.subplot(312)
                    #    #        plt.plot(ora_rVE2_time, ora_rVE2, 'g.', alpha=0.3)
                    #    #        plt.hold(True)
                    #    ##        plt.plot(inst_rVE2_time, inst_rVE2_smooth, 'g-', linewidth=1.5, alpha=0.9)
                    #    #        plt.plot(ora_rVE2_time, ora_rVE2_smooth, 'g-', linewidth=1.5)
                    #    #        plt.title('breath-by-breath Minute Volume -- VALLEY-TO-VALLEY',fontsize=13)
                    #    #        plt.legend(('%i/%i Avg' %(br_win-br_reject, br_win),'Smoothed %i/%i Avg' %(br_win-br_reject, br_win)), loc='upper left', fontsize=10)
                    #    #        plt.ylabel('rVE',fontsize=12)
                    #    #        plt.xticks(xticks,xlabels,rotation='vertical')
                    #    #        plt.grid(True)
                    #    #        plt.xlim(0,xticks[-1])
                    #    #
                    #    #        plt.subplot(313)
                    #    #        plt.plot(rVE2_time, rVE2, 'b.', alpha=0.1)
                    #    #        plt.plot(rVE2_time, rVE2_smooth, 'b-', linewidth=1.5)
                    #    #        plt.title('sec-by-sec Minute Volume (%isec-window) -- VALLEY-TO-VALLEY' % (win/fs),fontsize=13)
                    #    #        plt.legend(('Inst.','Smoothed'), loc='upper left', fontsize=10)
                    #    #        plt.ylabel('rVE',fontsize=12)
                    #    #        plt.xlabel('Time [mm:ss]',fontsize=12)
                    #    #        plt.xticks(xticks,xlabels,rotation='vertical')
                    #    #        plt.grid(True)
                    #    #        plt.xlim(0,xticks[-1])
                    #    #
                    #    #        plt.show()
                    #            # ----------------------------------------

                    # PEAK-TO-PEAK:
                    #        inst_br = np.transpose(np.vstack((inst_br_time, inst_br)))
                    ora_br = np.transpose(np.vstack((br_time, br, br_smooth)))
                    #        inst_rVE = np.transpose(np.vstack((inst_rVE_time, inst_rVE, inst_rVE_smooth)))
                    #        ora_rVE = np.transpose(np.vstack((ora_rVE_time, ora_rVE, ora_rVE_smooth)))
                    #        win_rVE = np.transpose(np.vstack((rVE_time, rVE, rVE_smooth)))
                    inst_rVT_raw = np.transpose(np.vstack((inst_rVE_time_raw, delta_amp_raw, delta_amp_raw_smooth,
                                                           delta_amp_raw / fev1, delta_amp_raw_smooth / fev1)))
                    inst_rVE_raw = np.transpose(np.vstack((inst_rVE_time_raw, inst_rVE_raw, inst_rVE_smooth_raw)))
                    win_rVE_raw = np.transpose(np.vstack((rVE_raw_time, rVE_raw, rVE_raw_smooth)))

                    # =============================================================================
                    # rVE as Multiple of Baseline:
                    #            '''
                    #            '''
                    #            VE = win_rVE_raw # <<< Select which VE signal to use for analysis and plot
                    #            '''
                    #            '''
                    #            if VE_baseline_thresh == []:
                    #                initial_rest_dur = json_data['info']['ird'] # [sec]
                    #                baseline_ind = np.argwhere((VE[:,0]<initial_rest_dur/60))[:,0]
                    #                # -15 since half of savgol filt win
                    #                baseline = np.mean(VE[baseline_ind,2]) # -15 since half of savgol filt win
                    #                if len(VE[baseline_ind,1]) >= 45:
                    #                    baseline = np.mean(savgol_filter(VE[baseline_ind,1],45,1))
                    #                VE_baseline_thresh = 1.1*baseline
                    #            VE_baseline_multiple = VE[:,2]/VE_baseline_thresh
                    #            VE_baseline_multiple[VE_baseline_multiple<1] = 1
                    #            VE_baseline_multiple = np.transpose(np.vstack((VE[:,0],VE_baseline_multiple)))
                    #
                    #            if VT_baseline_thresh == []:
                    #                initial_rest_dur = json_data['info']['ird'] # [sec]
                    #                baseline_ind = np.argwhere((inst_rVT_raw[:,0]<initial_rest_dur/60))[:,0]
                    #                # -15 since half of savgol filt win
                    #                baseline = np.mean(inst_rVT_raw[baseline_ind,2]) # -15 since half of savgol filt win
                    #                if len(inst_rVT_raw[baseline_ind,1]) >= 45:
                    #                    baseline = np.mean(savgol_filter(inst_rVT_raw[baseline_ind,1],45,1))
                    #                VT_baseline_thresh = 1.1*baseline
                    # =============================================================================

                    # VALLEY=TO-VALLEY:
                    #        inst_br = np.transpose(np.vstack((inst_br2_time, inst_br2)))
                    #        ora_br = np.transpose(np.vstack((br2_time, br2)))
                    #        inst_rVT = np.transpose(np.vstack((inst_rVE2_time, delta_amp)))
                    #        inst_rVE = np.transpose(np.vstack((inst_rVE2_time, inst_rVE2, inst_rVE2_smooth)))
                    #        ora_rVE = np.transpose(np.vstack((ora_rVE2_time, ora_rVE2, ora_rVE2_smooth)))
                    #        win_rVE = np.transpose(np.vstack((rVE2_time, rVE2, rVE2_smooth)))

                    #            inst_temp = np.empty(inst_rVE.shape,dtype=np.float)
                    #            ora_temp = np.empty(ora_rVE.shape,dtype=np.float)
                    #
                    #            sqi_resp = np.ones((1,8),dtype=np.float)

                    # Internal Load
                    ve = win_rVE_raw  # inst_rVE_raw # VE_baseline_multiple ## <<< Select VE signal to use
                    ve_sig = ve[:, 2]  # ve[:,1] ##- VE_baseline_thresh
                    ve_sig[ve_sig < 0] = 0
                    internal_load = np.nansum(ve_sig) / 1000  # *2 #/100
                    #        internal_load_signal = np.zeros((len(ve_sig),1))
                    #        for i in range(0,len(ve_sig)):
                    #            internal_load_signal[i] = sum(ve_sig[:i+1])
                    internal_load_signal = np.nancumsum(ve_sig * 10)
                    internal_load_signal = np.transpose(np.vstack((ve[:, 0], internal_load_signal)))

                    # Inhale:Exhale duration ratio
                    in_ = valley_peak_ALL[1:, 1] - valley_peak_ALL[1:, 0]  # peak(i) - valley(i) ## valley_1 >>> peak_1
                    ex_ = valley_peak_ALL[1:, 0] - valley_peak_ALL[:-1,
                                                   1]  # valley(i+1) - peak(i)  ## peak_1 >>> valley_2
                    in_ex = in_[1:] / ex_[1:]
                    in_ex[np.isnan(inst_br[:-1])] = np.nan

                    #                    try:
                    #                        in_ex[ind_nan+br_win-br_reject:ind_nan+br_win-br_reject+2]=np.nan
                    #                    except:
                    #                        pass
                    if len(in_ex) >= 15:
                        in_ex_smooth = savgol_filter(in_ex, 15, 1)
                    else:
                        in_ex_smooth = in_ex
                    in_ex_ratio = np.transpose(np.vstack((inst_br_time[:-1], in_ex, in_ex_smooth)))

                    #                    fl = 12
                    #                    fh = 6
                    #
                    #                    fig,axs = plt.subplots(figsize=(fl,fh),sharex=True)
                    #                    axs.plot(ora_br[:,0],ora_br[:,2],'m-',alpha=0.75,lw=1.5)
                    #                    axs.plot(ora_br[:,0],ora_br[:,1],'m.',alpha=0.2)
                    #                    axs.set_ylabel(('BR'),fontsize=12)
                    #                    axs.set_xlabel(('Time [$min$]'),fontsize=12)
                    #                    axs.set_xlim(0,ora_br[-1,0])
                    #                    axs.grid(True)
                    #                    ax2 = axs.twinx()
                    #                    ax2.plot(in_ex_ratio[:,0],in_ex_ratio[:,2],'-',color='slateblue',alpha=0.75,lw=1.5)
                    #                    ax2.plot(in_ex_ratio[:,0],in_ex_ratio[:,1],'.',color='slateblue',alpha=0.2)
                    #                    ax2.set_ylabel(('In/Ex'),fontsize=12)
                    #                    ax2.set_xlim(0,in_ex_ratio[-1,0])
                    ##                    ax2.set_ylim(0,2)

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

    except:
        ora_br = np.empty((0, 3))
        inst_rVT_raw = np.empty((0, 5))
        inst_rVE_raw = np.empty((0, 3))
        win_rVE_raw = np.empty((0, 3))
        internal_load = np.nan
        internal_load_signal = np.empty((0, 2))
        valley_peak_ALL = np.empty((0, 2))
        in_ex_ratio = np.empty((0, 2))
        X_bbyb = []
    return X_bbyb, ora_br, inst_rVT_raw, inst_rVE_raw, win_rVE_raw, internal_load, internal_load_signal, valley_peak_ALL, in_ex_ratio
