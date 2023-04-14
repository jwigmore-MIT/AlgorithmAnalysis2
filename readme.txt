# VE, VT, RRAvg Visualization
For Minute Volume, Tidal Volume, and Average Breathing rate visualization run
    > python vis1.py PATH/ACTIVITY_DIR/uncleaned_data

where PATH/ACTIVITY_DIR/ is a directory for an activity that contains both:
 1. The Raw JSON  = raw data + live algorithm outputs
 2. CSV = post-processing algorithm(s) outputs
which can all be obtained via exporting from dashboard-v2.tymewear.com

EXAMPLE:
    > python vis1.py data/Juan_2023-03-18_Testing_Live_VE_and_Garmin

NOTE: The above assumes the ACTIVITY_DIR is organized as:
    |--ACTIVITY_DIR
        |-- uncleaned_data
            |--RAW.json
            |--POSTPROCESSED.csv

running this will produce a new directory 'cleaned_data' in PATH/ACTIVITY_DIR/ contained pickled dataframes that are used in the analysis.
For example after running
    > python vis1.py PATH/ACTIVITY_DIR/uncleaned_data
The ACTIVITY_DIR should be structured as follows:
    |-- ACTIVITY_DIR
          |-- uncleaned_data
              |--RAW.json
              |--POSTPROCESSED.csv
          |-- cleaned_data
              |--aws_b3_df.pkl
              | ...
              |--raw_slow_df.pkl