from Error_Compare import *
import numpy as np


COLOR_DICT = {"chest": '#1f77b4', # muted blue
              "VE": '#ff7f0e', # safety orange,
              "LIVE": '#2ca02c', # cooked asparagus green
              "": '#19D3F3', # brick red,
              "PP": '#9467bd', # muted purple
              "ERROR": 'red' } # red

LINE_DICT = {"LIVE" : {"color": COLOR_DICT["LIVE"], "dash": "solid"},
             "PP": {"color": COLOR_DICT["PP"], "dash": "solid"},
             "CHEST": {"color": COLOR_DICT["chest"], "dash": "solid"},
             "ERROR": {"color": COLOR_DICT["ERROR"], "dash": "solid"}}


def ptp_abs_error(live, pp):
    # Live will be the warped data aligned to the x_times of pp
    error = np.zeros([len(live),2])
    for i in range(len(live)):
        x = live[i,0]
        # get pp_val at x
        pp_val = pp[pp[:,0] == x,1]
        error[i,:] = x,abs(live[i,1] - pp_val)
    return error

def ptp_percent_error(live, pp):
    # Live will be the warped data aligned to the x_times of pp
    error = np.zeros([len(live),2])
    for i in range(len(live)):
        x = live[i,0]
        # get pp_val at x
        pp_val = pp[pp[:,0] == x,1]
        error[i,:] = x,abs(live[i,1] - pp_val)/min(pp_val, live[i,1])
        # fill any nan values with last value
        error[np.isnan(error)] = error[np.isnan(error).sum(axis = 1) > 0,-1]
    return error

def get_frac_in_error(error_signal, low, high = np.inf):
    error_indices = error_signal[(error_signal[:,1] > low) & (error_signal[:,1] < high),0]
    #error_values = error_signal[(error_signal[:,1] > low) & (error_signal[:,1] < high),1]
    frac_in_error = len(error_indices)/len(error_signal)
    return frac_in_error, error_indices #error_values

def ptp_missed_peak_error(live, pp):
    error = np.zeros([len(live),2])
    for i in range(len(live)):
        x = live[i,0]
        # get pp_val at x
        pp_val = pp[pp[:,0] == x,1]
        error[i,:] = x, abs(live[i,1] - pp_val)/live[i,1]
        # fill any nan values with last value
        error[np.isnan(error)] = error[np.isnan(error).sum(axis = 1) > 0,-1]
    return error

def distance(q: np.ndarray,r:np.ndarray):

    return np.linalg.norm(q-r)

def return_path(acc_cost_mat):
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, sz2 - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = np.array([acc_cost_mat[i - 1][j - 1],
                               acc_cost_mat[i - 1][j],
                               acc_cost_mat[i][j - 1]])
            argmin = np.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]

def return_path2(D):
    N, M = D.shape
    path = np.zeros([N,2], dtype=int)
    for i in range(N):
        if i == 0:
            j = int(np.argmin(D[i,:]))
        else:
            min_j = path[i-1,1] + 1
            j = int(np.argmin(D[i, min_j:]))+ min_j
        path[i,:] = i, j

    return path






def custom_dtw(qry, ref, window = 5):
    D = np.ones([len(qry), len(ref)])*np.inf
    for i in range(len(qry)):
        max_rj0 = qry[i,0]
        # get closest ref less than max_rj0
        j_max = np.argmin(abs(ref[:,0] - max_rj0))
        # correct if time-stamp of ref[j_max] is greater than qry[i,0]
        if ref[j_max,0] > max_rj0:
            j_max -= 1
        j_min = max(0, j_max - window)
        # Compute DTW for all ref[j_min:j_max]
        for j in range(j_min, j_max):
            dist = distance(qry[i], ref[j])
            if i == 0:
                D[i,j] = dist
            else:
                D[i,j] = dist + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    # get min cost warping path
    path = return_path2(D)


    return D, path

def apply_path(path, query, ref):

    for i in range(len(path)):
        query[i,0] = ref[path[i,1],0]

    return query

def apply_path2(path: np.ndarray, query_df, ref_df):
    # reindex query_df based on the path
    query_df = query_df.iloc[path[:,1]]
    return query_df

def compute_live_error(live, pp):
    # first column is time stamp, second is the measure
    # for each time in live, compute the error for the measurement in pp
    # return the error for each time in live
    live_error = np.zeros([len(live),1])
    for i in range(len(live)):
        # find the closest time in pp
        j = np.argmin(abs(pp[:,0] - live[i,0]))
        live_error[i] = abs(live[i,1] - pp[j,1])
    return live_error

def compute_lower_bound_error(qry: np.ndarray, ref:np.ndarray, window = 4):
    # for each time t in qry, compare ref of t to qry of [t-window, t+window]
    # return the error for each time in live
    qry_error = np.zeros([len(qry),1])
    percent_error = np.zeros([len(qry),1])
    for i in range(len(qry)):
        # find the closest time in pp
        j_min = max(0, i - window)
        j_max = min(len(ref), i + window)
        j = np.argmin(abs(ref[j_min:j_max] - qry[i]))
        qry_error[i] = max(0,(qry[i] - ref[j_min + j]))
        percent_error[i] = qry_error[i]/ref[j_min + j]
    return qry_error, percent_error


def summarize_percent_error(error:np.ndarray, threshold = 0.5, error_cdf = False):
    results = {}
    results['mean'] = np.mean(error)
    results['median'] = np.median(error)
    results['std'] = np.std(error)
    results['min'] = np.min(error)
    results['max'] = np.max(error)
    results[f'PosRel (> {threshold}) [%]'] = np.sum(error > threshold) / len(error)
    results[f'PosRel (> {threshold}) [s]'] = np.sum(error > threshold)
    if error_cdf:
        # find fraction of time error is greater than each 10% of the range
        for i in range(1,10):
            time_in_error = np.sum(error > i * .1)
            results[f"{i*.1:.1f}% error time"] = time_in_error
            results[f"{i*.1:.1f}% error frac"] = (time_in_error / len(error)).round(3)
    return results

def compute_RAT_error(error_signal, error_threshold, window): #"Rolling Average Threshold" (RAT) error
    # compute the fraction of points in error_signal that are greater than error_threshold within the window
    # starting from point i to [i-window, i+window]
    # return the rolling average error signal
    RA_error = np.zeros([len(error_signal),1])
    for i in range(len(error_signal)):
        j_min = max(0, i - window)
        j_max = min(len(error_signal), i + window)
        RA_error[i] = np.sum(error_signal[j_min:j_max] > error_threshold)/len(error_signal[j_min:j_max])
    return RA_error


def check_breathing_gaps(df, max_gap = 10):
    results = {}
    # compute the breathTimeDiff
    df['breathTimeDiff'] = df['breathTime'].diff()
    # check if the max breathTimeDiff is greater than max_gap
    results['breathTimeDiff'] = df['breathTimeDiff'].max() > max_gap

def method1_error_analysis(live_b3_df, pp_b3_df, metric, return_all = False, breath_gap = 10, min_time = 60, max_time = -60, live_shift = -6, error_window = 3, error_threshold = 0.5, RAT_error_window = 30, RAT_error_threshold = 0.5):

    results = {}

    # Run cleaning process on breath-by-breath data
    pp_s_df = run_cleaning_process(pp_b3_df)
    live_s_df = run_cleaning_process(live_b3_df)

    # Select the time range of interest
    if max_time <0: # if given max time is negative, subtract it from the min max time index
        max_time = min(pp_s_df.index.max(), live_s_df.index.max()) + max_time
    pp_s_df = pp_s_df.loc[(pp_s_df.index >= min_time) & (pp_s_df.index <= max_time)]
    live_s_df = live_s_df.loc[(live_s_df.index >= min_time) & (live_s_df.index <= max_time)]

    # Shift the live data based on the time shift
    live_s_df.index = live_s_df.index + live_shift

    # Get the time-ranges where both pp and live are valid
    min_s = max(min(pp_s_df.index), min(live_s_df.index))
    max_s = min(max(pp_s_df.index), max(live_s_df.index))
    # Apply the time range to the two dataframes
    pp_s_df = pp_s_df.loc[(pp_s_df.index >= min_s) & (pp_s_df.index <= max_s)]
    live_s_df = live_s_df.loc[(live_s_df.index >= min_s) & (live_s_df.index <= max_s)]

    # Get the percent positive relative error
    pp_error, pp_percent_error = compute_lower_bound_error(pp_s_df[metric].to_numpy(), live_s_df[metric].to_numpy(), error_window)
    live_error, live_percent_error = compute_lower_bound_error(live_s_df[metric].to_numpy(), pp_s_df[metric].to_numpy(), error_window)

    pp_error_summary = summarize_percent_error(pp_percent_error, error_threshold)
    live_error_summary = summarize_percent_error(live_percent_error, error_threshold)

    # Compute a rolling average of the fraction of time each percent error exceeds the threshold
    pp_RAT_error = compute_RAT_error(pp_percent_error, error_threshold, RAT_error_window)
    live_RAT_error = compute_RAT_error(live_percent_error, error_threshold, RAT_error_window)

    # check if the RAT error exceeds the threshold
    pp_RAT_flag = np.sum(pp_RAT_error > RAT_error_threshold) > 0
    live_RAT_flag = np.sum(live_RAT_error > RAT_error_threshold) > 0

    pp_error_summary[f"RAT flag ({RAT_error_window}s/{RAT_error_threshold})"] = pp_RAT_flag
    live_error_summary[f"RAT flag ({RAT_error_window}s/{RAT_error_threshold})"] = live_RAT_flag

    results.update({"pp_error_summary": pp_error_summary, "live_error_summary": live_error_summary})
    if return_all:
        all_signals = {
            "pp_s_df": pp_s_df,
            "live_s_df": live_s_df,
            "pp_error": pp_error,
            "pp_percent_error": pp_percent_error,
            "live_error": live_error,
            "live_percent_error": live_percent_error,
            "pp_RAT_error": pp_RAT_error,
            "live_RAT_error": live_RAT_error,
        }
        results["all_signals"] = all_signals

    return results

def plot_all_signals(raw_chest, results, metric, dataset = ""):
    all_signals = results["all_signals"]
    live_s_df = all_signals["live_s_df"]
    pp_s_df = all_signals["pp_s_df"]
    pp_percent_error = all_signals["pp_percent_error"]
    live_percent_error = all_signals["live_percent_error"]
    pp_RAT_error = all_signals["pp_RAT_error"]
    live_RAT_error = all_signals["live_RAT_error"]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=(
    "Raw Chest Signal", "Input Signals", "PP to Live LB Relative Percent Error",
    "Live to PP LB Relative Percent Error"))
    fig.update_layout(title_text=f" {metric} Method 1 Error Analysis for {dataset}")

    # Plot the raw chest signal
    fig.add_trace(go.Scatter(x=raw_chest["time"], y=raw_chest["c"], name="Raw Chest Signal", line=LINE_DICT["CHEST"]),
                  row=1, col=1)
    fig.update_yaxes(title_text="Chest Signal", row=1, col=1)

    # Plot the VT Signals (interpolated and live shifted in this case)
    fig.add_trace(go.Scatter(x=live_s_df.index, y=live_s_df[metric], name="Live (-6s)", line=LINE_DICT["LIVE"]), row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=pp_s_df.index, y=pp_s_df[metric], name="PP (s)", line=LINE_DICT["PP"]), row=2, col=1)
    fig.update_yaxes(title_text=metric, row=2, col=1)

    # plot the error
    fig.add_trace(go.Bar(x=pp_s_df.index, y=pp_percent_error[:, 0], name="PP to Live  PosRel Error",
                         marker_color=COLOR_DICT["ERROR"]), row=3, col=1)
    fig.add_trace(go.Scatter(x=pp_s_df.index, y=pp_RAT_error[:, 0], name="PP to Live RAT Error", line=LINE_DICT["ERROR"]),
                  row=3, col=1)
    fig.update_yaxes(title_text="Percent Positive Error", row=3, col=1)
    fig.update_yaxes(range=[0, 1], row=3, col=1)
    # Add annotations for the error summary
    pp_error_summary = results["pp_error_summary"]
    pp_error_summary_str = ""
    for key, value in pp_error_summary.items():
        if value in ["min", "max", "std", "median"]:
            pass
        else:
            if isinstance(value, float):
                value = round(value, 3)
            pp_error_summary_str += f"{key}: {value}<br>"
    # place annotation in subplot 3
    fig.add_annotation(x=0.5, y=1.1, xref="paper", yref="paper", text=pp_error_summary_str, showarrow=False, font=dict(
        size=10), align="left", row=3, col=1)




    # plot the error
    fig.add_trace(go.Bar(x=live_s_df.index, y=live_percent_error[:, 0], name="Live to PP PosRel Error",
                         marker_color=COLOR_DICT["ERROR"]), row=4, col=1)
    fig.add_trace(
        go.Scatter(x=live_s_df.index, y=live_RAT_error[:, 0], name="Live to PP RA Error", line=LINE_DICT["ERROR"]),
        row=4, col=1)
    fig.update_yaxes(title_text="Percent Positive Error", row=4, col=1)
    fig.update_yaxes(range=[0, 1], row=4, col=1)
    # Add annotations for the error summary
    live_error_summary = results["live_error_summary"]
    live_error_summary_str = ""
    for key, value in live_error_summary.items():
        if value in ["min", "max", "std", "median"]:
            pass
        else:
            if isinstance(value, float):
                value = round(value, 3)
            live_error_summary_str += f"{key}: {value}<br>"
    # place annotation in subplot 3
    fig.add_annotation(x=0.5, y=1.1, xref="paper", yref="paper", text=live_error_summary_str, showarrow=False, font=dict(
        size=10), align="left", row=4, col=1)

    fig.update_xaxes(title_text="Time (s)", row=4, col=1)

    fig.show()





    # Compute error summary






