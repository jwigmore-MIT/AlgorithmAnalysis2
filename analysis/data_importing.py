import json
import os
import pickle as pkl

import pandas as pd

AWS_NAME_MAP_2_LIVE = {'Breath by breath time': 'breathTime',
                       'BR breath by breath': 'instBR',
                       'BR breath by breath outlier rejected average': 'RRAvg',
                       # 'VT breath by breath smoothed' : 'VT_smoothed',
                       'VT breath by breath': 'VT',
                       'VE breath by breath smoothed': 'VE',
                       }


def get_raw_from_json(raw_json_path, reorder=True, process=True):
    raw = json.load(open(raw_json_path, 'rb'))
    if not process:
        return raw
    # Processing samples (Raw Data)
    samples = raw['samples']
    fast_data = {}  # Data sampled at 125Hz
    slow_data = {}  # Data sampled at 25Hz
    is_fast = {}
    for measurement, data in samples[0].items():
        if isinstance(data, list):
            fast_data[measurement] = []
            is_fast[measurement] = True
        elif isinstance(data, (int, float)):
            slow_data[measurement] = []
            is_fast[measurement] = False
    for sample in samples:
        for measurement, data in sample.items():
            if is_fast[measurement]:
                fast_data[measurement].extend(data)
            else:
                slow_data[measurement].append(data)

    # Get current time-stamps
    f_fast = 125  # Hz
    f_slow = 25  # Hz
    t_fast = [x / f_fast for x in range(len(fast_data['ax']))]
    t_slow = [x / f_slow for x in range(len(slow_data['id']))]
    fast_data['time'] = t_fast
    slow_data['time'] = t_slow

    fast_df = pd.DataFrame(fast_data)
    fast_cols = fast_df.columns.to_list()
    fast_cols = fast_cols[-1:] + fast_cols[:-1]
    fast_df = fast_df[fast_cols]

    slow_df = pd.DataFrame(slow_data)
    slow_cols = slow_df.columns.to_list()
    slow_cols = slow_cols[-1:] + slow_cols[:-1]
    slow_df = slow_df[slow_cols]

    ## Processing 'rtBreathing' (Live Algo Output)
    rtBreathing = raw["rtBreathing"]
    if len(rtBreathing) > 0:
        rtData = {}
        is_v37 = raw["info"]['fw_v'] == '0.37'
        if is_v37:
            Exception("Data is from Live Algorithm 0.37 -- Need to divide metrics by 100")
        for measurement, data in rtBreathing[0].items():
            rtData[measurement] = []
        for data_dict in rtBreathing:
            for measurement, data in data_dict.items():
                rtData[measurement].append(int(data))
        rt_df = pd.DataFrame(rtData)
        if reorder:
            order = [name for name in AWS_NAME_MAP_2_LIVE.values()]
            order.extend([name for name in rt_df.columns if name not in order])
            rt_df = rt_df[order]
    else:
        print("")
        rt_df = None

    return fast_df, slow_df, rt_df


def get_raw_from_dir_path(dir_path):
    '''

    :param dir_path:
    :return: raw
    '''
    json_file_paths = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.json'):
            json_file_paths.append(os.path.join(dir_path, file_name))
    if len(json_file_paths) > 1:
        print("RECHECK UNCLEANED DATA DIRECTORY - MULTIPLE JSONS FOUND")
    else:
        raw_json_path = json_file_paths[0]
    raw = get_raw_from_json(raw_json_path, process=False)
    return raw


def find_header(csv_path):
    """
    csv headers can be different sizes. This will identify
    the line that starts with 'Time' which begins the tabular data
    :param csv_path:
    :return: number of rows to skip
    """
    skip_rows = 0
    with open(csv_path, "r+") as f:
        for line in f:
            if not line.startswith("Time"):
                skip_rows += 1
            else:
                break
    return skip_rows


def get_chest_at_breathTime(raw_slow_df, b3_df):
    '''
    For each 'breathTime' time stamp in b3_df, gets the raw chest signal at the time stamp
    from the raw_slow_df Dataframe

    :param raw_slow_df: Dataframe containing the raw breath signal
    :param b3_df: Dataframe containing breath by breath measurements (e.g. matlab_b3_df, aws_b3_df)
    :return: new_b3_df : Original b3_df with second column "c" as the raw chest data at the corresponding "breathTime"
    '''
    b3_cols = list(b3_df.columns)
    b3_cols.insert(1, "c")
    merged = pd.merge(b3_df, raw_slow_df, left_on="breathTime", right_on="time", sort=False, indicator=True)
    new_b3_df = merged.loc[:, b3_cols]

    return new_b3_df


def get_breathTime_diff(b3_df):
    """
    Computes the difference between consecutive "breathTime" timestamps and sets the values
    in the third column of the input b3 Dataframe
    :param b3_df: Breath by breath dataframe
    :return: Modified b3_df with difference of consecutive breathTimes in the second column
    """
    # Compute difference between consecutive columns
    b3_df["breathTimeDiff"] = b3_df["breathTime"].diff()

    # Reorder b3_df to have the breathTimeDiff in the third column
    cols = list(b3_df.columns)
    cols.insert(2, cols.pop(cols.index("breathTimeDiff")))
    b3_df = b3_df.loc[:, cols]
    return b3_df


def get_aws_data_from_csv(proc_csv_path, rename=True, reorder=True, round_breath_time=True):
    # Define the CSV file path and specify that the header starts at row 17
    csv_path = proc_csv_path
    skip_rows = find_header(csv_path)  # index starts at 0, so this is the 17th row

    # Read in the CSV file with Pandas, skipping the first 17 rows and specifying the column names
    df = pd.read_csv(csv_path, skiprows=skip_rows, header=None, names=list(range(0, 27)))

    # Rename the columns based on the names in row 18
    col_names = df.iloc[0].tolist()

    nan_idx = next((i for i, v in enumerate(col_names) if v != v),
                   -1)  # this is the column that splits the data between x_label = seconds and x_label = breath_by_breath
    unit_names = df.iloc[1].tolist()
    df = df.iloc[2:]

    # Set the column names
    # df.columns = pd.MultiIndex.from_tuples(zip(col_names[:28], unit_names))
    df.columns = col_names

    time_data = df.iloc[:, :nan_idx]
    aws_b3_df = df.iloc[:, nan_idx + 1:]

    # Remove all rows from the interval_data where Breath by breath is nan
    aws_b3_df = aws_b3_df.dropna(how='all')

    # Remove all columns from interval data that are nan
    aws_b3_df = aws_b3_df.dropna(axis='columns', how='all')

    # Convert all strings to floats
    aws_b3_df = aws_b3_df.astype(float)

    # Change names
    if rename:
        aws_b3_df.columns = [AWS_NAME_MAP_2_LIVE[name] if name in AWS_NAME_MAP_2_LIVE.keys() else name for name in
                             aws_b3_df.columns]
    # Reorder
    if rename and reorder:
        order = [name for name in AWS_NAME_MAP_2_LIVE.values()]
        order.extend([name for name in aws_b3_df.columns if name not in order])
        aws_b3_df = aws_b3_df[order]

    if round_breath_time:
        aws_b3_df["breathTime"] = aws_b3_df["breathTime"].round(decimals=2)

    return time_data, aws_b3_df


def clean_all_data(uncleaned_data_dir):
    def pickle_all(cleaned_data_dir, df_dict):
        for df_name, df in df_dict.items():
            file_path = os.path.join(cleaned_data_dir, df_name + '.pkl')
            pkl.dump(df, open(file_path, 'wb'))

    ## FILE OPERATION
    # get path of raw json within dir
    json_file_paths = []
    for file_name in os.listdir(uncleaned_data_dir):
        if file_name.endswith('.json'):
            json_file_paths.append(os.path.join(uncleaned_data_dir, file_name))
    if len(json_file_paths) > 1:
        print("RECHECK UNCLEANED DATA DIRECTORY - MULTIPLE JSONS FOUND")
    else:
        raw_json_path = json_file_paths[0]

    # get path of processed (AWS) csv path
    csv_file_paths = []
    for file_name in os.listdir(uncleaned_data_dir):
        if file_name.endswith('.csv'):
            csv_file_paths.append(os.path.join(uncleaned_data_dir, file_name))
    if len(csv_file_paths) > 1:
        print("RECHECK UNCLEANED DATA DIRECTORY - MULTIPLE CSVs FOUND")
    else:
        proc_csv_path = csv_file_paths[0]
    # Creating clean_data directory if needed
    parent_dir = os.path.dirname(uncleaned_data_dir)
    cleaned_folder_path = os.path.join(parent_dir, 'cleaned_data')
    if not os.path.exists(cleaned_folder_path):
        os.mkdir(cleaned_folder_path)
        print(f"Created cleaned_data folder at {cleaned_folder_path}")
    else:
        print(f"cleaned_data folder already exists at {cleaned_folder_path}")

    dd = {}
    # Get the raw 125Hz data (fast_df), raw 25Hz data (slow_df), and real-time breath-by-breath data (rt_b3_df) from the "{}_Raw.csv"
    dd["raw_fast_df"], dd["raw_slow_df"], dd["live_b3_df"] = get_raw_from_json(raw_json_path)

    # Get the processed time-dependent data and the breath-by-breath dependent data from the "Processed data-{}.csv"
    dd["aws_time_df"], aws_b3_df = get_aws_data_from_csv(proc_csv_path)

    ## Get raw chest for each breath detection time
    # AWS
    aws_b3_df = get_chest_at_breathTime(dd["raw_slow_df"], aws_b3_df)
    dd["aws_b3_df"] = get_breathTime_diff(aws_b3_df)

    pickle_all(cleaned_folder_path, dd)


def load_cleaned_data(cleaned_data_dir):
    cleaned_data = {}
    for file_name in os.listdir(cleaned_data_dir):
        if file_name.endswith('.pkl'):
            cleaned_data[os.path.splitext(file_name)[0]] = pkl.load(
                open(os.path.join(cleaned_data_dir, file_name), 'rb'))
    return cleaned_data
