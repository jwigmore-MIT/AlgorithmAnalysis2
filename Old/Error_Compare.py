import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from interfaces.postprocessing import BR_rVE_RTformat_wrapper_peak_detection as pp_peak_detection
import analysis.data_importing as imp  # Custom importing module
import analysis.plotting as pl  # Custom plotting module
import interfaces.postprocessing as pif  # post-processing interface
import scipy.signal
import warnings
import os
import scipy
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import dtw  # Dynamic Time Warping
#import external.custom_post.custom_post_copy as cpc
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.plotting.backend = "plotly"



COLOR_DICT = {"chest": '#1f77b4', # muted blue
              "VE": '#ff7f0e', # safety orange,
                "VT": '#2ca02c', # cooked asparagus green
              "RRAvg": '#19D3F3', # brick red,
                "instBR": '#9467bd',} # muted purple

LINE_DICT = {"LIVE" : {"VE" : {"color": COLOR_DICT["VE"], "dash": "solid"},
                       "VT" : {"color": COLOR_DICT["VT"], "dash": "solid"},
                       "RRAvg" : {"color": COLOR_DICT["RRAvg"], "dash": "solid"},
                          "instBR" : {"color": COLOR_DICT["instBR"], "dash": "solid"}},
             "PP": {"VE" : {"color": COLOR_DICT["VE"], "dash": "dot"},
                    "VT" : {"color": COLOR_DICT["VT"], "dash": "dot"},
                    "RRAvg" : {"color": COLOR_DICT["RRAvg"], "dash": "dot"},
                    "instBR" : {"color": COLOR_DICT["instBR"], "dash": "dot"},}}

# DATA IMPORTING
def load_data(activity_data_dir, has_uncleaned = True):
    if has_uncleaned:
        uncleaned_data_dir = os.path.join(activity_data_dir, "uncleaned_data")
    else:
        uncleaned_data_dir = activity_data_dir

    cleaned_data_dir = activity_data_dir
    # check if the activity data directory contains pickled data:
    pkl_names = ["aws_b3_df.pkl", "aws_time_df.pkl", "live_b3_df.pkl", "raw_slow_df.pkl"]
    clean_dfs = {}
    for name in pkl_names:
        if not os.path.exists(os.path.join(uncleaned_data_dir, name)):
            print("No pickled data found in activity data directory. Cleaning data...")
            clean_dfs = imp.clean_all_data(uncleaned_data_dir, create_dir=True)
            break
        # if not, clean the data and save it
    if clean_dfs == {}:
        print("Pickled data found in activity data directory. Loading data...")
        clean_dfs = imp.load_cleaned_data(uncleaned_data_dir)
    return clean_dfs

# DATA CLEANING
def run_cleaning_process(b3_df, index = "breathTime", demo = False):
    og_df = deepcopy(b3_df)
    # Cast to int
    as_int = cast_to_int(og_df)
    # Set index to breathTime

    as_int = as_int.set_index(index)
    og_df = og_df.set_index(index)
    # Interpolate to second by second resolution
    min_index, max_index = as_int.index.min(), as_int.index.max()
    sec = pd.DataFrame(np.arange(min_index, max_index), index = np.arange(min_index, max_index))
    joined = as_int.join(sec, how = "outer")
    interpolated = joined.interpolate(method = "index").drop(columns=[0])
    # check if interpolated contains nans
    if interpolated.isnull().values.any():
        print("Warning: interpolated data contains NaN values. Backfilling...")
        backfilled = interpolated.fillna('backfill')
    else: backfilled = interpolated
    # Average over all repeated indices (caused by breaths falling within the same second)
    grouped = backfilled.groupby(backfilled.index).mean().astype(int)
    if demo: # return all intermediate dataframes
        return grouped, dict(og_df = og_df, as_int = as_int, joined = joined, interpolated = interpolated, backfilled = backfilled, grouped = grouped)
    else:
        return grouped

def cast_to_int(df):
    for col in df.columns:
        df[col] = df[col].fillna(method = 'backfill').astype(int)
    return df
# def b3_to_seconds(b3_df, make_int = True, index = "breathTime", demo = False):
#     # Interpolate data frame to second by second resolution
#     new_df = deepcopy(b3_df)
#     if demo:
#         df_dicts = OrderedDict()
#         df_dicts["precleaning"] = deepcopy(new_df)
#     if make_int:
#         new_df = cast_to_int(new_df)
#         if demo: df_dicts["as_int"] = deepcopy(new_df)
#             # if indices repeat, take average of value at that index
#     new_df = new_df.set_index(index)
#     min_index, max_index = new_df.index.min(), new_df.index.max()
#     sec = pd.DataFrame(np.arange(min_index, max_index), index = np.arange(min_index, max_index))
#     new_df = new_df.join(sec, how = "outer")
#     new_df_interp = new_df.interpolate(method = "index").drop(columns=[0])
#     if demo: df_dicts["interpolated"] = deepcopy(new_df_interp)
#     new_df_filled = new_df_interp.fillna('backfill')
#     if demo: df_dicts["backfilled"] = deepcopy(new_df_filled)
#     new_df_grouped = new_df_filled.groupby(new_df.index).mean().astype(int)
#     if demo: df_dicts["grouped"] = deepcopy(new_df_grouped)
#     if demo: return new_df, df_dicts
#
#     return new_df

# PLOTTING


def plot_metrics(chest_df, metric_df, data_set: str):
    n = len(metric_df.columns)+1
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    # Chest
    fig.add_trace(go.Scatter(x=chest_df["time"], y=chest_df["c"], name="Chest"), row=1, col=1)
    # Metrics
    for i, col in enumerate(metric_df.columns):
        fig.add_trace(go.Scatter(x=metric_df.index, y=metric_df[col], name=col, line =dict(color =  COLOR_DICT[col])), row=i+2, col=1)
    fig.update_layout(title_text=f"{data_set} Metrics")
    fig.show()
    return fig
def plot_metrics_compare(chest_df, live_df, pp_df, show = False, errors_indices: dict = None):
    n = len(live_df.columns)+1
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=["Raw Chest Data"]+list(live_df.columns))
    # Chest
    fig.add_trace(go.Scatter(x=chest_df["time"], y=chest_df["c"], name="Chest"), row=1, col=1)
    # Metrics
    for i, col in enumerate(live_df.columns):
        fig.add_trace(go.Scatter(x=live_df.index, y=live_df[col], name=f"Live {col}", line = LINE_DICT["LIVE"][col]), row=i+2, col=1)
        fig.add_trace(go.Scatter(x=pp_df.index, y=pp_df[col], name=f"PP {col}", line = LINE_DICT["PP"][col]), row=i+2, col=1)
    if errors_indices is not None:
        for metric, error_indices in errors_indices.items():
            y_val = live_df[metric].loc[error_indices]
            fig.add_trace(go.Scatter(x=y_val.index, y=y_val, name=f"Live {metric} Error", mode="markers", marker=dict(color="red", size=5)), row=live_df.columns.get_loc(metric)+2, col=1)


    fig.update_layout(title_text="Live vs Post-Processed Metrics",
                      hovermode="x unified",)
    fig.update_xaxes(title_text="Time (s)", row=n, col=1)
    if show:
        fig.show()
    return fig

def plot_metric_error(raw_chest,live_df, pp_df, metric, error_signal, error_indices):
    fig = make_subplots(rows = 3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    # Chest
    fig.add_trace(go.Scatter(x=raw_chest["time"], y=raw_chest["c"], name="Chest"), row=1, col=1)
    # Metrics
    fig.add_trace(go.Scatter(x=live_df.index, y=live_df[metric], name=f"Live {metric}", line = LINE_DICT["LIVE"][metric]), row=2, col=1)
    fig.add_trace(go.Scatter(x=pp_df.index, y=pp_df[metric], name=f"PP {metric}", line = LINE_DICT["PP"][metric]), row=2, col=1)
    # Error Signal
    fig.add_trace(go.Scatter(x=error_signal.index, y=error_signal, name="Error Signal", line = dict(color = COLOR_DICT[metric])), row=3, col=1)
    # Error Indices
    fig.add_trace(go.Scatter(x=error_indices, y=error_signal[error_indices], name="Error Indices", mode="markers", marker=dict(color="red")), row=3, col=1)
    fig.update_layout(title_text=f"Error Indices for {metric}",
                        hovermode="x unified",)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.show()

def plot_error_indices(raw_chest, error_signals, error_indices_dict):
    fig = make_subplots(rows = 2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    # Chest
    fig.add_trace(go.Scatter(x=raw_chest["time"], y=raw_chest["c"], name="Chest"), row=1, col=1)
    # Error Indices
    bars = []
    for metric in error_indices_dict.keys():
        # make dataframe where error indices are 1 and everything else is 0
        error_indices = error_indices_dict[metric]
        error_signal = error_signals[metric]
        ind_error_signal = pd.Series([1 if i in error_indices else 0 for i in error_signal.index], index = error_signal.index)
        fig.add_trace(go.Bar(x=ind_error_signal.index, y=ind_error_signal, name=f"{metric}", marker_color = COLOR_DICT[metric]), row=2, col=1)

    fig.update_layout(barmode='stack', bargap = 0)
    # plot bars
    #fig.add_trace(go.Bar(x=bars_df.index, y=[x for x in bars_df.columns], name="HR Error Indices"), row=2, col=1)
    fig.update_layout(title_text=f"Error Indices for {metric}",
                        hovermode="x unified",)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.show()





# CROSS CORRELATION

def get_corr_lag(live_df, pp_df, metric, plot = False):
    corr= scipy.signal.correlate(live_df[metric], pp_df[metric])
    lags = scipy.signal.correlation_lags(len(live_df[metric]), len(pp_df[metric]))
    lag = lags[np.argmax(corr)]
    if plot:
        fig = make_subplots(rows = 1, cols=1)
        fig.add_trace(go.Scatter(x=lags, y=corr, name="Correlation", line = dict(color = COLOR_DICT[metric])), row=1, col=1)
        fig.update_layout(title_text=f"Cross Correlation of {metric} Live vs Post-Processed")
        fig.show()
    return lag, corr


def shift_signal(signal: pd.Series, shift: int):
    '''
    Shifts the signal by the given shift
    :param signal: metric data (e.g. tidal volume)
    :param shift: amount to shift the time-series by (+ right, - left)
    :return:
    '''
    shifted_signal = deepcopy(signal)
    shifted_signal.index = shifted_signal.index + shift
    return shifted_signal

def gen_error_signal(live_signal, pp_signal, error_type = "Percent"):
    '''
    Generates the error signal between the live and post-processed signals
    :param live_signal: live signal
    :param pp_signal: post-processed signal
    :param error_type: type of error to calculate
    :return: error signal
    '''
    if error_type == "Absolute":
        error_signal = np.abs(live_signal - pp_signal)
    if error_type == "Squared":
        error_signal = (live_signal - pp_signal)**2
    if error_type == "Percent":
        # percent error = abs(live - pp)/min(live, pp)
        error_signal = np.abs(live_signal - pp_signal)/np.minimum(live_signal, pp_signal)
    return error_signal

def get_time_in_error(error_signal, low, high = np.inf):
    '''
    Calculates the time the error signal is between the low threshold and high threshold
    :param error_signal: error signal
    :param high: high threshold
    :param low: low threshold
    :return: time in error, error_indices
    '''

    error_indices = error_signal.index[np.where((error_signal >= low) & (error_signal <= high))[0]]
    time_in_error = len(error_indices)
    return time_in_error, error_indices

def get_error_durations(error_indices):
    error_durations = []
    start_error_ind = []
    end_error_ind = []
    i = 0
    while i < len(error_indices):
        index = error_indices[i]
        if index not in start_error_ind:
            start_error_ind.append(index)
        j = i + 1
        while j < len(error_indices):
            if error_indices[j] == error_indices[j-1] + 1:
                j += 1
            else:
                break
        end_error_ind.append(error_indices[j-1])
        error_durations.append(error_indices[j-1] - error_indices[i] + 1)
        i = j
    duration_df = pd.DataFrame({"start": start_error_ind, "end": end_error_ind, "duration": error_durations})
    return duration_df


def corr_error(live_df, pp_df, metric, shift, error_type = "MAE", plot = False):
    # shift live_df by shift
    live_df_sh = deepcopy(live_df)
    live_df_sh[metric] = live_df[metric].shift(shift)
    # get error
    if error_type == "MAE":
        error = np.abs(live_df_sh[metric] - pp_df[metric])
        mean_error = np.mean(error)
    if error_type == "MSE":
        error = (live_df_sh[metric] - pp_df[metric])**2
        mean_error = np.mean(error)

    if plot:
        fig = make_subplots(rows = 2, cols=1, shared_xaxes= True, subplot_titles=[metric, error_type])
        fig.add_trace(go.Scatter(x=live_df_sh.index, y=live_df_sh[metric], name="Live", line = LINE_DICT["LIVE"][metric]), row=1, col=1)
        fig.add_trace(go.Scatter(x=pp_df.index, y=pp_df[metric], name="Post-Processed", line = LINE_DICT["PP"][metric]), row=1, col=1)
        # plot MSE
        fig.add_trace(go.Scatter(x=live_df_sh.index, y=error, name= error_type, line = dict(color = COLOR_DICT[metric])), row=2, col=1)
        fig.update_layout(title_text=f"{error_type} of {metric} Live vs Post-Processed")
        fig.show()
    else:
        fig = None
    return mean_error, fig

def perform_error_analysis(live_df, pp_df, key_metrics, lags = "None", error_type = "Percent", lower_threshold = 0.2, upper_threshold = np.inf):

    if lags is None:
        lags = {metric: 0 for metric in key_metrics if metric != "breathTime"}
    all_time_in_error = {}
    all_error_indices = {}
    all_error_signals = {}
    all_shifted_series = {}
    all_error_fraction = {}
    all_error_durations = {}
    for metric in key_metrics:
        if metric == "breathTime":
            continue
        # shift live_df by shift
        series = live_df[metric]
        shifted_series = shift_signal(series, lags[metric])
        all_shifted_series[metric] = shifted_series
        # get error
        error = gen_error_signal(shifted_series, pp_df[metric], error_type=error_type)
        # get time in error
        time_in_error, error_indices = get_time_in_error(error, lower_threshold, upper_threshold)

        all_error_durations[metric] = get_error_durations(error_indices)
        error_fraction = time_in_error/len(error)

        #print(f"{metric} time in error: {time_in_error}")
        #print(f"{metric} error fraction: {error_fraction}")
        all_error_signals[metric] = error
        all_time_in_error[metric] = time_in_error
        all_error_indices[metric] = error_indices
        all_error_fraction[metric] = error_fraction
    live_df_sh = pd.DataFrame(all_shifted_series)
    error_signals = pd.DataFrame(all_error_signals)


    return all_error_signals, all_time_in_error, all_error_indices, all_error_fraction, all_error_durations, live_df_sh



    

if __name__ == '__main__':

    activity_dir = "data/effad897-2991-4c2c-9a6a-f85a111d0e3d"
    clean_dfs = load_data(activity_dir)

    min_time = 60
    max_time = np.inf

    lower_threshold = .20 # for error
    upper_threshold = np.inf # for error
    error_type = "Percent"


    # Load Data
    raw_chest = clean_dfs["raw_slow_df"][["time","c",]]
    key_metrics = ["breathTime", "VE", "VT"]# "VT", "RRAvg", "instBR"]
    live_b3_df = clean_dfs["live_b3_df"][key_metrics]
    pp_b3_df = clean_dfs["aws_b3_df"][key_metrics]

    # Only look at a subset of the data
    raw_chest = raw_chest[(raw_chest["time"] > min_time) & (raw_chest["time"] < max_time)]
    live_b3_df = live_b3_df[(live_b3_df["breathTime"] > min_time) & (live_b3_df["breathTime"] < max_time)]
    pp_b3_df = pp_b3_df[(pp_b3_df["breathTime"] > min_time) & (pp_b3_df["breathTime"] < max_time)]

    live_df = run_cleaning_process(live_b3_df[key_metrics])
    pp_df, clean_demo = run_cleaning_process(pp_b3_df[key_metrics], demo = True)



    # Get Cross Correlation Lags
    # lag_limit = 6
    # lags = {}
    # for metric in key_metrics:
    #     if metric == "breathTime":
    #         continue
    #     lag, corr = get_corr_lag(live_df, pp_df, metric, plot = False)
    #     print(f"{metric} lag: {lag}")
    #     lags[metric] = lag

    all_time_in_error = {}
    all_error_indices = {}
    all_error_signals = {}
    all_shifted_series = {}
    all_error_fraction = {}
    all_error_durations = {}

    lag = 0
    for metric in key_metrics:
        if metric == "breathTime":
            continue
        # shift live_df by shift
        series = live_df[metric]
        shifted_series = shift_signal(series, lag)
        all_shifted_series[metric] = shifted_series
        # get error
        error = gen_error_signal(shifted_series, pp_df[metric], error_type=error_type)
        # get time in error
        time_in_error, error_indices = get_time_in_error(error, lower_threshold, upper_threshold)

        all_error_durations[metric] = get_error_durations(error_indices)
        error_fraction = time_in_error/len(error)

        print(f"{metric} time in error: {time_in_error}")
        print(f"{metric} error fraction: {error_fraction}")
        all_error_signals[metric] = error
        all_time_in_error[metric] = time_in_error
        all_error_indices[metric] = error_indices
        all_error_fraction[metric] = error_fraction
    live_df_sh = pd.DataFrame(all_shifted_series)
    metric= "VE"
    plot_metric_error(raw_chest, live_df_sh, pp_df, metric, all_error_signals[metric], all_error_indices[metric])
    fig = plot_metrics_compare(raw_chest, live_df_sh, pp_df, all_error_indices)




