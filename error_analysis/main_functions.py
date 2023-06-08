import numpy as np
from data_retrieval import run_cleaning_process
from plotly.subplots import make_subplots
import plotly.graph_objects as go




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


def compute_RAT_error(error_signal, error_threshold, window): #"Rolling Average Threshold" (RAT) error
    '''
    Compute the rolling average error signal which characterizes the frequency of significant errors within set windows
    :param error_signal: typically the percent error signal
    :param error_threshold: set threshold that determines what is a significant error
    :param window: half the window size to compute the rolling average error signal
    :return: Rolling average error signal
    '''
    RA_error = np.zeros([len(error_signal),1])
    for i in range(len(error_signal)):
        j_min = max(0, i - window)
        j_max = min(len(error_signal), i + window)
        RA_error[i] = np.sum(error_signal[j_min:j_max] > error_threshold)/len(error_signal[j_min:j_max])
    return RA_error

def compute_error(qry: np.ndarray, ref:np.ndarray, window = 4):
    """
    For each time in qry, compare ref of t to qry of [t-window, t+window]
    :param qry: query signal
    :param ref: "ground truth" signal
    :param window: half-window size to compare the qry(t) to ref(t-window, t+window)
    :return: the error for each time in qry
    """
    qry_error = np.zeros([len(qry),1])
    percent_error = np.zeros([len(qry),1])
    for i in range(len(qry)):
        j_min = max(0, i - window)
        j_max = min(len(ref), i + window)
        j = np.argmin(abs(ref[j_min:j_max] - qry[i]))
        qry_error[i] = max(0,(qry[i] - ref[j_min + j]))
        percent_error[i] = qry_error[i]/ref[j_min + j]
    return qry_error, percent_error

def summarize_percent_error(error:np.ndarray, threshold = 0.5, error_cdf = False):
    """
    For an error signal, compute statistics, time in error, and fraction of time in error
    :param error: error signal (example percent error)
    :param threshold: the threshold to consider for time in error and fraction of time in error
    :param error_cdf: if true, will return the fraction of time in error for each 10% of the range
    :return:
    """
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

def method1_error_analysis(live_b3_df, pp_b3_df, metric, return_all = False, min_time = 60, max_time = -60, live_shift = -6, error_window = 3, error_threshold = 0.5, RAT_error_window = 30, RAT_error_threshold = 0.5):
    """

    :param live_b3_df: live breath-by-breath dataframe
    :param pp_b3_df: post-processed breath-by-breath dataframe
    :param metric: VT, VE, instBR, etc.
    :param return_all: if True, return all the cleaned and error dataframes, if False, return only the summary
    :param min_time: minimum time (s) to consider in both series
    :param max_time: maximum time (s) to consider in both series
    :param live_shift: amount to shift the live data by (s)
    :param error_window: window (s) to consider for error analysis
    :param error_threshold: threshold to consider for error analysis
    :param RAT_error_window: window (s) to consider for RAT error analysis
    :param RAT_error_threshold: threshold to consider for RAT error analysis
    :return: dataframe of summary statistics (and optionally the cleaned and error dataframes)


    """
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

    # Get the error (*_error) and the percent positive relative error (*_percent_error)
    pp_error, pp_percent_error = compute_error(pp_s_df[metric].to_numpy(), live_s_df[metric].to_numpy(), error_window)
    live_error, live_percent_error = compute_error(live_s_df[metric].to_numpy(), pp_s_df[metric].to_numpy(),
                                                   error_window)
    # Create a dataframe of the error signals (recommend to use percent error)
    pp_error_summary = summarize_percent_error(pp_percent_error, error_threshold)
    live_error_summary = summarize_percent_error(live_percent_error, error_threshold)

    # Compute a rolling average of the fraction of time each percent error exceeds the threshold
    pp_RAT_error = compute_RAT_error(pp_percent_error, error_threshold, RAT_error_window)
    live_RAT_error = compute_RAT_error(live_percent_error, error_threshold, RAT_error_window)

    # check if the RAT error exceeds the threshold
    pp_RAT_flag = np.sum(pp_RAT_error > RAT_error_threshold) > 0 # if any of the RAT error exceeds the threshold, flag it
    live_RAT_flag = np.sum(live_RAT_error > RAT_error_threshold) > 0 # if any of the RAT error exceeds the threshold, flag it
    pp_error_summary[f"RAT flag ({RAT_error_window}s/{RAT_error_threshold})"] = pp_RAT_flag
    live_error_summary[f"RAT flag ({RAT_error_window}s/{RAT_error_threshold})"] = live_RAT_flag

    # Add all results to the results dictionary
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

    fig.show(renderer="browser")