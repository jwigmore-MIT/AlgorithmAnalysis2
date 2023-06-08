from interfaces.postprocessing import *
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

json_dir_path = "/Old/data\dataset\c26fb5ae-9c5d-423b-b4e7-ff30a41db2f9"
chest_raw, chest_5hz, chest_bs, chest_bs_smooth, time, X_bbyb_df = BR_rVE_RTformat_wrapper(json_dir_path)

columns = ['Breath by breath time', 'BR breath by breath', 'BR breath by breath outlier rejected average', 'VT breath by breath',
        'VT breath by breath smoothed', 'VE breath by breath', 'VE breath by breath smoothed',
        'Inhale:Exhale Ratio breath by breath', 'Inhale:Exhale Ratio breath by breath smoothed']
X_bbyb_df.columns = columns
units = ['sec', 'br/min', 'br/min', 'volume/br',
        'volume/br', 'volume/min', 'volume/min', 'sec/sec', 'sec/sec']

# write the data in X_bbyb_df to a csv file, where the first two rows are the column names and units
X_bbyb_df.to_csv('C:\GitHub\AlgorithmAnalysis2\data\dataset\c26fb5ae-9c5d-423b-b4e7-ff30a41db2f9\c26fb5ae-9c5d-423b-b4e7-ff30a41db2f9.csv', header=columns, index=False)

# Interpolate BR, VT, and VE to a 1 Hz sampling rate
s_df = X_bbyb_df[["Breath by breath time", 'BR breath by breath outlier rejected average', 'VT breath by breath smoothed', 'VE breath by breath smoothed']]
columns = ["breathTime", "BR", "VT", "VE"]
s_df.columns = columns
s_df["breathTime"] = s_df["breathTime"].astype(int)
s_df = s_df.set_index("breathTime")
min_index, max_index = s_df.index.min(), s_df.index.max()
sec = pd.DataFrame(np.arange(min_index, max_index), index=np.arange(min_index, max_index))
s_df = s_df.join(sec, how="outer")
s_df = s_df.interpolate(method="index").drop(columns=[0])
# round each value to nearest 2 decimal places
s_df = s_df.round(2)
# make the index the first column
s_df.reset_index(level=0, inplace=True)
s_df.columns = columns
s_df.to_csv('C:\GitHub\AlgorithmAnalysis2\data\dataset\c26fb5ae-9c5d-423b-b4e7-ff30a41db2f9\second_to_second.csv', header= ["Time", "BR", "VT", "VE"], index=False)

if True:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=("BR", "VT", "VE"))

        fig.add_trace(go.Scatter(x=s_df["breathTime"], y=s_df["BR"], name="BR"), row=1, col=1)
        fig.add_trace(go.Scatter(x = X_bbyb_df["Breath by breath time"], y = X_bbyb_df["BR breath by breath outlier rejected average"], name="BR"), row=1, col=1)

        fig.add_trace(go.Scatter(x=s_df["breathTime"], y=s_df["VT"], name="VT"), row=2, col=1)
        fig.add_trace(go.Scatter(x = X_bbyb_df["Breath by breath time"], y = X_bbyb_df["VT breath by breath smoothed"], name="VT"), row=2, col=1)

        fig.add_trace(go.Scatter(x=s_df["breathTime"], y=s_df["VE"], name="VE"), row=3, col=1)
        fig.add_trace(go.Scatter(x = X_bbyb_df["Breath by breath time"], y = X_bbyb_df["VE breath by breath smoothed"], name="VE"), row=3, col=1)

        fig.show()