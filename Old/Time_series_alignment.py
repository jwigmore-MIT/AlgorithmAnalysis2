import pandas as pd

from Error_Compare import *
import analysis.dtw as cdtw
import dtw as pdtw
import numpy as np
# https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html
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





## Load the Data
# Timeframe
min_time = 120
live_min_time = 130 # we want a slightly delayed start for the live data
max_time = np.inf
# Key Metrics
key_metrics = ["breathTime", "VT", "instBR"]#, "VT", "RRAvg", "instBR"]

# Get and Clean Data
activity_dir = "data/dataset/2b8f256d-7063-410c-b66d-3bdfb7d140c5"

activity = os.path.split(activity_dir)[-1]
clean_dfs = load_data(activity_dir, has_uncleaned = False)

# Load Data
raw_chest = clean_dfs["raw_slow_df"][["time","c",]]
live_b3_df = clean_dfs["live_b3_df"][key_metrics]
pp_b3_df = clean_dfs["aws_b3_df"][key_metrics]

# Apply timeframe constraints
raw_chest = raw_chest[(raw_chest["time"] > min_time) & (raw_chest["time"] < max_time)]
live_b3_df = live_b3_df[(live_b3_df["breathTime"] > live_min_time) & (live_b3_df["breathTime"] < max_time)]
pp_b3_df = pp_b3_df[(pp_b3_df["breathTime"] > min_time) & (pp_b3_df["breathTime"] < max_time)]
# Backfill nans if necessary
if pp_b3_df.isnull().values.any():
    print("pp_b3_df CONTAINS NaNs- backfilling")
    pp_b3_df = pp_b3_df.fillna(method = "bfill")

## Cast pp_b3_df as int
pp_b3_df = pp_b3_df.astype(int)

## Add in time-difference between consecutive breaths for live_b3_df and pp_b3_df
# live_b3_df["prevBreathTimeDiff"] = live_b3_df["breathTime"].diff()
# live_b3_df["prevBreathTimeDiff"] = live_b3_df["prevBreathTimeDiff"].fillna(0)
# pp_b3_df["prevBreathTimeDiff"] = pp_b3_df["breathTime"].diff()
# pp_b3_df["prevBreathTimeDiff"] = pp_b3_df["prevBreathTimeDiff"].fillna(0)


# Normalize each metric
norm_factors: dict = {}
live_b3_df_n = live_b3_df.copy()
pp_b3_df_n = pp_b3_df.copy()
for metric in live_b3_df.columns:
    max_val = max(live_b3_df[metric].max(), pp_b3_df[metric].max())
    min_val = min(live_b3_df[metric].min(), pp_b3_df[metric].min())
    norm_factors[metric] = {"max": max_val, "min": min_val}
    # Apply normalization
    live_b3_df_n[metric] = (live_b3_df[metric] - min_val)/(max_val - min_val)
    pp_b3_df_n[metric] = (pp_b3_df[metric] - min_val)/(max_val - min_val)

##
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
    for i in range(len(qry)):
        # find the closest time in pp
        j_min = max(0, i - window)
        j_max = min(len(ref), i + window)
        j = np.argmin(abs(ref[j_min:j_max] - qry[i]))
        qry_error[i] = abs(qry[i] - ref[j_min + j])
    percent_error = qry_error/qry
    return qry_error, percent_error




qry = live_b3_df_n.to_numpy()
ref = pp_b3_df_n.to_numpy()

D, path =  custom_dtw(qry, ref, window = 5)

og_live = live_b3_df.to_numpy()
og_pp = pp_b3_df.to_numpy()
live_warped = apply_path(path, deepcopy(og_live), og_pp)
# Custom DTW
# DTW Settings
window = [1,10]
gamma = 2 # scaling of shifting deviation penalty
beta = 0.2 # exponential averaging factor

# Original Lag only DTW method
VT_DTW, mapping2, _ = cdtw.dtw_distance(og_live, og_pp, avg_shift= True, window = window, gamma = gamma, beta = beta)
live_warped2 = cdtw.remap_query(og_live, og_pp, mapping2)
mapping3 = cdtw.fix_i_to_j(mapping2, live_warped2)
live_warped3 = cdtw.remap_query(live_warped2, og_pp, mapping3)

## Final Warping
# shift all of live to the left by the average shift
# avg_shift = np.mean(live_warped3[:,0] - og_live[:,0])
# og_live_shifted = deepcopy(og_live)
# og_live_shifted[:,0] = og_live[:,0] - avg_shift
# # use DTW to find the best warping
# DTW = pdtw.dtw(og_live_shifted, og_pp[:,1], keep_internals=True, step_pattern="asymmetric", open_end = True, open_begin=True)


# Compute Error
## Cannt easily compute error on non-time-aligned signal

## Warp 1 Error
# First we need to get them on the same x-axis

error1  = compute_live_error(live_warped[:,:2], og_pp[:,:2])
sum_error1 = np.sum(error1)

error2 = compute_live_error(live_warped2, og_pp[:,:2])
sum_error2 = np.sum(error2)

error3 = compute_live_error(live_warped3, og_pp[:,:2])
sum_error3 = np.sum(error3)

## Lets try resampled version
import matplotlib.pyplot as plt
pp_s_df = run_cleaning_process(clean_dfs["aws_b3_df"][key_metrics])
live_s_df = run_cleaning_process(clean_dfs["live_b3_df"][key_metrics])
# select only where index is between min_time and max_time
pp_s_df = pp_s_df.loc[(pp_s_df.index >= min_time) & (pp_s_df.index <= max_time)]
live_s_df = live_s_df.loc[(live_s_df.index >= min_time) & (live_s_df.index <= max_time)]
# shift all live indices by -6
live_s_df.index = live_s_df.index - 6
# remove all live values with index less than the min index of pp
live_s_df = live_s_df.loc[live_s_df.index >= pp_s_df.index[0]]
# remove all pp values with index greater than the max index of live
pp_s_df= pp_s_df.loc[pp_s_df.index <= live_s_df.index[-1]]
# Use DTW to find the optimal path from live to pp
live_s_VT = live_s_df["VT"].to_numpy()
pp_s_VT = pp_s_df["VT"].to_numpy()
alignment = pdtw.dtw(live_s_VT, pp_s_VT, keep_internals=True, step_pattern="asymmetric", open_end = True, open_begin=True)
# plot the warped live and pp
ax = alignment.plot(type="twoway")
fig = ax.get_figure()
fig.show()
# Map
map_ = np.array([alignment.index1, alignment.index2])


# apply the mapping to the live data
live_s_VT_warped = apply_path2(map_.T, live_s_df, pp_s_df)
# plot the two series
if True:
    fig = make_subplots(rows = 2, cols = 1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=live_s_df.index, y=live_s_df["VT"], name="Live", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pp_s_df.index, y=pp_s_df["VT"], name="PP", mode="lines"), row=1, col=1)

    fig.add_trace(go.Scatter(x=pp_s_df.index, y=pp_s_df["VT"], name="PP", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=live_s_VT_warped.index, y=live_s_VT_warped["VT"], name="Live Warped", mode="lines"), row=2, col=1)

    fig.show()

## Now try with taking derivaties of the inputs
pp_s_df = run_cleaning_process(clean_dfs["aws_b3_df"][key_metrics])
live_s_df = run_cleaning_process(clean_dfs["live_b3_df"][key_metrics])
# select only where index is between min_time and max_time
pp_s_df = pp_s_df.loc[(pp_s_df.index >= min_time) & (pp_s_df.index <= max_time)]
live_s_df = live_s_df.loc[(live_s_df.index >= min_time) & (live_s_df.index <= max_time)]
# shift all live indices by -6
live_s_df.index = live_s_df.index - 6
# remove all live values with index less than the min index of pp
live_s_df = live_s_df.loc[live_s_df.index >= pp_s_df.index[0]]
# remove all pp values with index greater than the max index of live
pp_s_df= pp_s_df.loc[pp_s_df.index <= live_s_df.index[-1]]
# Use DTW to find the optimal path from live to pp
live_s_VT = live_s_df["VT"].to_numpy()
pp_s_VT = pp_s_df["VT"].to_numpy()
# compute the derivatives
live_s_VT_d = np.diff(np.diff(live_s_VT))
pp_s_VT_d = np.diff(np.diff(pp_s_VT))
# compute the alignment
alignment = pdtw.dtw(pp_s_VT_d, live_s_VT_d, keep_internals=True, step_pattern="asymmetric", open_end = True, open_begin=True)
# get the mapping of indices from live to pp

map_ = np.array([alignment.index1, alignment.index2])
# apply the mapping to the live data
live_s_VT_warped = apply_path2(map_.T, live_s_df, pp_s_df)
# resample the live data
live_s_VT_warped = live_s_VT_warped.groupby(live_s_VT_warped.index).mean()
# interpolate such that index is increasing by 1
live_s_VT_warped = live_s_VT_warped.reindex(np.arange(live_s_VT_warped.index[0], live_s_VT_warped.index[-1]+1))
# interpolate the missing values
live_s_VT_warped = live_s_VT_warped.interpolate(method="linear")
# take the intersection of the indices
live_s_VT_warped = live_s_VT_warped.loc[live_s_VT_warped.index.isin(pp_s_df.index)]
pp_s_df = pp_s_df.loc[pp_s_df.index.isin(live_s_VT_warped.index)]
# Compute lower bound pp_s_df to live_s_VT_warped error
pp_lb_error, _ = compute_lower_bound_error(pp_s_df["VT"].to_numpy(), live_s_VT_warped["VT"].to_numpy())
# Compute the lower bound of the live_s_VT_warped to pp_s_df error
live_lb_error, _ = compute_lower_bound_error(live_s_VT_warped["VT"].to_numpy(), pp_s_df["VT"].to_numpy())

# plot signals and the error
if True:
    fig = make_subplots(rows = 3, cols = 1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=live_s_VT_warped.index, y=live_s_VT_warped["VT"], name="Live", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pp_s_df.index, y=pp_s_df["VT"], name="PP", mode="lines"), row=1, col=1)

    # plot the error
    fig.add_trace(go.Bar(x=pp_s_df.index, y=pp_lb_error[:,0], name="PP to Live  LB Error", marker_color="red"), row=2, col=1)

    # plot the error
    fig.add_trace(go.Bar(x = live_s_VT_warped.index, y=live_lb_error[:,0], name="Live to PP LB Error", marker_color="blue"), row=3, col=1)

    fig.show()

## Now try just by shifting by an the estimated time difference
pp_s_df = run_cleaning_process(clean_dfs["aws_b3_df"][key_metrics])
live_s_df = run_cleaning_process(clean_dfs["live_b3_df"][key_metrics])
# select only where index is between min_time and max_time
pp_s_df = pp_s_df.loc[(pp_s_df.index >= min_time) & (pp_s_df.index <= max_time)]
live_s_df = live_s_df.loc[(live_s_df.index >= min_time) & (live_s_df.index <= max_time)]
# shift all live indices by -6
live_s_df.index = live_s_df.index - 6
# remove all live values with index less than the min index of pp
live_s_df = live_s_df.loc[live_s_df.index >= pp_s_df.index[0]]
# remove all pp values with index greater than the max index of live
pp_s_df= pp_s_df.loc[pp_s_df.index <= live_s_df.index[-1]]

# compute the error
pp_lb_error, pp_percent_error = compute_lower_bound_error(pp_s_df["VT"].to_numpy(), live_s_df["VT"].to_numpy())
# Compute the lower bound of the live_s_VT_warped to pp_s_df error
live_lb_error, live_percent_error = compute_lower_bound_error(live_s_df["VT"].to_numpy(), pp_s_df["VT"].to_numpy())

# plot signals and the error
if True:
    fig = make_subplots(rows = 3, cols = 1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=live_s_df.index, y=live_s_df["VT"], name="Live", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pp_s_df.index, y=pp_s_df["VT"], name="PP", mode="lines"), row=1, col=1)

    # plot the error
    fig.add_trace(go.Bar(x=pp_s_df.index, y=pp_percent_error[:,0], name="PP to Live  LB Error", marker_color="red"), row=2, col=1)

    # plot the error
    fig.add_trace(go.Bar(x = live_s_VT_warped.index, y=live_percent_error[:,0], name="Live to PP LB Error", marker_color="blue"), row=3, col=1)

    fig.show()



# merge the time series


# Compute error
#error = compute_live_error(live_s_VT_warped[["VT"]].to_numpy(), pp_s_df[["VT"]].to_numpy())
##
if True:
    fig = make_subplots(rows = 2, cols = 1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=live_s_df.index, y=live_s_df["VT"], name="Live", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pp_s_df.index, y=pp_s_df["VT"], name="PP", mode="lines"), row=1, col=1)

    fig.add_trace(go.Scatter(x=pp_s_df.index, y=pp_s_df["VT"], name="PP", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=live_s_VT_warped.index, y=live_s_VT_warped["VT"], name="Live Warped", mode="lines"), row=2, col=1)

    fig.show()

##


if False:
    # plot the two series
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=live_s_df.index, y=live_s_df["VT"], name="Live", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pp_s_df.index, y=pp_s_df["VT"], name="PP", mode="lines"), row=1, col=1)
    fig.show()


## Plotting
if False:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=live_warped[:,0], y=live_warped[:,1], name="Live Warped1", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pp_b3_df["breathTime"], y=pp_b3_df["VT"], name="PP", mode="lines"), row=1, col=1)

    # plot error as bar plot
    fig.add_trace(go.Bar(x=live_warped[:,0], y=error1[:,0], name="Error", marker_color="red"), row=2, col=1)


    fig.show()

##
if False:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=live_warped2[:, 0], y=live_warped2[:, 1], name="Live Warped2", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pp_b3_df["breathTime"], y=pp_b3_df["VT"], name="PP", mode="lines"), row=1, col=1)

    # plot error as bar plot
    fig.add_trace(go.Bar(x=live_warped2[:, 0], y=error2[:, 0], name="Error", marker_color="red"), row=2, col=1)

    fig.show()
##
if False:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=live_warped3[:, 0], y=live_warped3[:, 1], name="Live Warped3", mode="lines"), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=pp_b3_df["breathTime"], y=pp_b3_df["VT"], name="PP", mode="lines"), row=1, col=1)

    # plot error as bar plot
    fig.add_trace(go.Bar(x=live_warped3[:, 0], y=error3[:, 0], name="Error", marker_color="red"), row=2, col=1)

    fig.show()

##
if False:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=live_b3_df["breathTime"], y=live_b3_df["VT"], name="Live", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pp_b3_df["breathTime"], y=pp_b3_df["VT"], name="PP", mode="lines"), row=1, col=1)

    fig.add_trace(go.Scatter(x=pp_b3_df["breathTime"], y=pp_b3_df["VT"], name="PP", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=live_warped[:,0], y=live_warped[:,1], name="Live Warped", mode="lines"), row=2, col=1)


    fig.add_trace(go.Scatter(x=pp_b3_df["breathTime"], y=pp_b3_df["VT"], name="PP", mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x = live_warped2[:,0], y = live_warped2[:,1], name = "Live Warped2", mode = "lines"), row = 3, col = 1)


    fig.add_trace(go.Scatter(x=pp_b3_df["breathTime"], y=pp_b3_df["VT"], name="PP", mode="lines"), row=4, col=1)
    fig.add_trace(go.Scatter(x = live_warped3[:,0], y = live_warped3[:,1], name = "Live Warped2", mode = "lines"), row = 4, col = 1)

    fig.show()

## Look at instBR
if False:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=live_b3_df["breathTime"], y=live_b3_df["instBR"], name="Live", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pp_b3_df["breathTime"], y=pp_b3_df["instBR"], name="PP", mode="lines"), row=1, col=1)

    fig.add_trace(go.Scatter(x=pp_b3_df["breathTime"], y=pp_b3_df["instBR"], name="PP", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=live_warped[:,0], y=live_warped[:,2], name="Live Warped", mode="lines"), row=2, col=1)

    fig.show()









# ## VT DTW Settings
# VT_window1 = [1,10]
# VT_window2 = 5
# gamma = 2 # scaling of shifting deviation penalty
# beta = 0.2 # exponential averaging factor
#
# # First convert to numpy arrays
# VT_live = live_b3_df[["breathTime", "VT"]].to_numpy()
# VT_pp = pp_b3_df[["breathTime", "VT"]].to_numpy()
#
# VT_DTW, VT_DTW_index_map1, _ = dtw.dtw_distance(VT_live, VT_pp, avg_shift= True, window = VT_window1, gamma = gamma, beta = beta)
# VT_warped = dtw.remap_query(VT_live, VT_pp, VT_DTW_index_map1)
# #VT_i_to_j_fixed = dtw.fix_i_to_j(VT_DTW_index_map1, warped)
# VT_DTW2 = dtw.dtw2(VT_warped, VT_pp, VT_window2)
# path2 = dtw.get_warp_path(VT_DTW2)
# #VT_warped2 = dtw.remap_query(VT_warped, VT_pp, path2)
#
# ## Plot All signals
# fig = go.Figure()
# fig.add_trace(go.Scatter(x = VT_live[:,0], y = VT_live[:,1], name = "Live"))
# fig.add_trace(go.Scatter(x = VT_pp[:,0], y = VT_pp[:,1], name = "PP"))
# fig.add_trace(go.Scatter(x = VT_warped[:,0], y = VT_warped[:,1], name = "DTW"))
# #fig.add_trace(go.Scatter(x = VT_warped2[:,0], y = VT_warped2[:,1], name = "DTW2"))
#
