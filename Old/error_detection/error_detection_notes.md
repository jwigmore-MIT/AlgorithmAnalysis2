# Pipeline

**Main Steps**

1. Data Retrieval: How do we retrieve the required data for error analysis, and where do we store generated data
2. Data Preprocessing

## (1) Data Retrieval

**Required Data**:

1. Live algorithm outputs for VT, instBR, VE
2. Post processed data for VT, instBR, VE
3. (Optional) Raw chest data

### (1.1) Local File Storage

#### (1.1.1) Non-processed data storage (i.e. Live and PP)

For each activity, create a directory in 'data/datasets/' with the activities name.

#### (1.1.2) Data Retrieval 

**Live Outputs (Breath-to-breath (b3))**

Download the json output from the dashboard and store in activity folder

*Note:* Currently has to be the only .json in the activity folder for the `Error_compare.load(data)` function to work

**Post Processing Outputs**

(Currently) Download the csv from the dashboard and store in the activity folder

*Note*: Currently has to be the only .csv in the activity folder for the `Error_compare.load(data)` function to work

(Planned) Run the post-processing script locally to generate the outputs

#### (1.1.3) Processed Data storage (i.e. cleaned for error detection)

##### Processed Data Directory

In initial data cleaning process, create a directory in the activity directory for cleaned data

### (1.2) Remote Version

#### (1.2.1) Non-processed Storage

Live algorithm (and other raw data) is already stored on S3
_Question_: Is the breath-by-breath data stored on S3? What about the second by second data?

#### (1.2.2) Data Retrieval

**Live**: Some call to the S3 database to retrieve the .json

**Post Processing**: If already stored, call to S3 database to retrieve it

##### (1.2.3) Processed Data storage

*It is probably worth storing any processed data on the server*

## (2) Data Preprocessing

## (2.1) Overview of preprocessing

**Goal**: Standardize the outputs between the two sources

0. Check there are any outstanding issues with the raw data (e.g. large periods of no data, errors in the the chest data, etc) or with the live algorithm outputs (e.g. doesn't exist)
1. Rename the data (if using dataframes)
2. | New Name  | Live Header | PP Header             |

---

| breathTime| breathTime  | Breath by breath time |
|VT         |VT breath by breath smoothed         |
|instBR	    |BR breath by breath                  |
|RRAvg      |BR breath by breath outlier rejected average |
|VE         |VE breath by breath smoothed                 |

2. Interpolate all b3 data to a second-to-second (s2) timescale

Live: 

Post-processing: this data might already exist on the S3 server

## (a) Analysis

### (a.1) Single

### (a.2) Collection

## (b) Results Formating

## (c) Results Saving


# General Questions

1. How much of the pipeline should I be responsible for?

# Questions for backend

1. Should I avoid using DataFrames?
2. What is stored from the post-processing algorithm on s3?
   1. Breath-by-breath outputs?
   2. Second-to-second outouts?
3. json (or its data) retrieval from s3?
4. Any stored post-processing outputs stored on s3 retrieval?
