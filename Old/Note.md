## References

METRICS = ["VE", "VT", "instBR", "RRAvg"]

PP: Post-processing algorithm

LIVE: Live algorithm

# Main Components:

## Data Cleaning

**TODO**: Create a workbook that demonstrates all aspects of the datacleaning

### Post-Processing int-casting

**What:** For the PP data, converting the "breathTime" and all METRICS values to integers

**Why**: All LIVE metrics and breath-times are casted as integers for the json.  For a fair comparison, we do the same to the PP data.

### Converting to second-by-second timescale

**What:** The original timestamps for both algorithms correspond to breath-times. We take each timeseries (whose index corresponds with integer casted breathimes) and fill the indices such that each second is represented in the data.  The values for all the metrics at these new rows are NaNs.

**Example**

Original Dataframe


| breathTime | metric |
| ---------- | ------ |
| 64         | 12     |
| 67         | 9      |

New Dataframe


| breathTime | metric |
| ---------- | ------ |
| 64         | 12     |
| 65         | NaN    |
| 66         | NaN    |
| 67         | 9      |


### Interpolating over all seconds

**What**: Interpolate between all NaN values for each metric

**Example**

Interpolated Dataframe

| breathTime | metric |
| ---------- |--------|
| 64         | 12     |
| 65         | 11     |
| 66         | 10     |
| 67         | 9      |




# Compensating for time-lag

## Cross-Correlation based shifting

Result will be the shift that results in the minimum MSE over the entire series?

## Signal modifications

**Post-processing as integers**: converts all times and metrics from post-processing to integers

**Fill NaN**: All nans in each series are backfilled i.e. filled with the next non-nan value

## 

import pandas as pd

# Defne the series with the desired index values

index_series = pd.Series([2, 4, 6, 8], index=[1, 2, 3, 4])

# Define the series for which you want to check the index presence

check_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])

# Create a new series where the value is one if the index is in the other series, zero otherwise

result_series = pd.Series([1 if index in index_series.index else 0 for index in check_series.index], index=check_series.index)

# Print the result series

print(result_series)
