# Handoff Notes

## Useful code

## runner.py

Python script to run the error analysis and produce the plots

### Notes on assumed file structure

The current data retrieval function assumes the following directory structure:

-error_analysis
---dataset
------activity1   
---------activity1.json
---------activity1.csv
---------live.csv (optional)
------activity2
---------activity2.json
...

Where activityx.json is the json output from the Live algorithm and activityx.csv is the csv output from the Post-Processing algorithm. 





Below are details on the techniques used (and not used) to compute useful error metrics

## Steps for Error Analysis

1. Data Retrieval: will change for deployment on the backend
2. Data Processing: may change on the backend (less reliance on
3. Signal Processing:
4. Error Signal Computation
5. Error Detection/Statistics
6. Outputs




## Time-Series alignment

The main challenge with detecting the "error" between the Live and Post-Processing (PP) signals is that the signals are not aligned in time. Below are some details on the challenges and attempted methods to align the time-series, but to make things short: **Time-series aligment between two unevenly sampled series is fundementally a hard problem. Its much easier to find an approximate alignment (with some data interpolation) and then compute an approximate lower-bound on the error**

### Challenges

Finding the exact 1-to-1 matching between all measurements from the Live and the PP algorithms is a challenging problem because:

1. The Live is shifted by a non-constant amount that depends on the filter being applied in real-time.
   1. The breathTime for the Live is truncated as an integer further leading to uncertanties in the timing of the breath as well as differences between the consecutive breaths
2. Each algorithm may have false-positives (i.e. output a detected breath when no breath occured) or false-negatives (i.e. should've detected a breath but did not).  Thus there actually doesn't exist a perfect 1-to-1 matching between each measurement series.
3. We are working with non-uniformly sampled time-series (i.e. breaths, and therefor metric measurements, are output at uneven intervals).  Standard time-series alignment methods e.g. Dynamic Time Warping, typically assume that the datasets have even sampling intervals.

I was able to get close (i.e. align 99% of all measurements between the Live and PP algorithms) using a custom **Dynamic Time Warping Algorithm** (detailed below).

### Solution

**Main idea:**

1. Interpolate both the Live and PP time-series from breath-to-breath to second-by-second intervals.
2. Shift the Live time-series by a constant amount to approximate the best overall fit
   1. Empirically, shifting the Live time-series by -6 seconds works the best
   2. Can also compute the cross-correlation between the interpolated series to find the best constant shift, but found this really wasn't worth it
3. Compensate for the non-perfect alignment between the two-series when computing the error signals (detailed below)

### Dynamic Time Warping (Not used)

https://en.wikipedia.org/wiki/Dynamic_time_warping

Dynamic Time Warping (DTW) is a popular algorithm used to find the "optimal match" between two misaligned time-series. It essentially computes a distance measure between each point in the two time-series, and then finds the optimal matching to minimize the sum of distances.

The standard DTW algorithm (and most variants) assume the time-series are sampled at the same constant intervals.  Ours are not and this is additionally complicated by the false-positive/false-negative breath detection problem as described above.

I developed an alignment algorith very similar to DTW that for each point $x(t)$ in the Live Series, it would compute the distance between all PP data points within some window $[t-w,w]$, and then find the mapping to minimize the cumulative distance.  However, the "spiky" nature of our time-series would lead to poor alignments. I tried to use derivative based methods so the algorithm would do a better job at aligning peaks to peaks and valleys to valleys (see https://www.ics.uci.edu/~pazzani/Publications/sdm01.pdf), however I was never able to come up with a method that would work 100% of the time for all time-series.

The reason I am including notes on DTW is just to bring this algorithm to your attention as it could be useful for future improvements to peak-detection or event-detection algorithms.  Here are a couple references that may be useful:

https://link.springer.com/article/10.1007/s10618-015-0418-x

https://towardsdatascience.com/time-series-classification-using-dynamic-time-warping-61dcd9e143f6

# Errors

There are two main steps in detecting errors:

1. Creating error signal(s) i.e. an estimate of the error at each point in time between two signals
2. Detecting significant error events i.e. using the error signal(s) to detect if there are large or frequent errors between the Live and PP algorithms

## Error Computation

Since the Live and PP time-series are not perfectly aligned, we need to compensate for this when computing the error. The main idea is to compute the *distance* between a point in one series to surrounding points in the other series and take the minimum distance as a lower-bound of the true error.

Let the "query" signal $\mathbf q$ and "reference" signal $\mathbf r$ be two second-to-second time-series to be compared.For example, $\mathbf q$ is the interpolated and shifted Live VT series and $\mathbf r$ is the interpolated PP VT series. Let $w$ be the "window" parameter.

The error at time $t$ is equal to the minimum distance between $q(t)$ and a measurement in $[r(t-w), r(t+1-w), ..., r(t), r(t+1), ... r(t+w)]$. This is essentially a lower bound on the error between the query and reference that must be used since we don't know the true alignment between the two signals.

The sudo code is below:

```sudo
for each t in T
    t_min = t-w
    t_max = t+w
    t* = argmin(|q(t)-r(t')| for t in [t_min, t_max]
    error(t) = max(0, q(t)-r(t*))
    percent_error(t) = error(t)/r(t*)
```

**Why max(0, q(t)-r(t\*))?**

We restrict the error signal to be non-negative as this method is meant to detect periods where $\mathbf q$ is much larger than $\mathbf r$. If $r(t)$ is much larger than $q(t^*)$ this signifies we either have a falsely detected breath in $\mathbf q$ or a missed breath in $\mathbf r$.  Additionally, I recommend switching what signal is the query and what is the reference. For example, compute the error signal with the Live as the query the PP as the reference, and then compute the error with the PP as the query and the Live as the reference.  This will help detect false-positives and false-negatives in both signals.

**Percent Error**

The percent error signal may be more useful for error classification/identification as it is normalized by the reference signal.  The scales between different metrics (e.g. VT vs instBR) may be very different. Additionally, there may be variance in the scale of metrics between different users.  Normalizing by the reference signal helps to account for these differences.
A percent error of 0.25 signifies the error is 25% of the reference signal.  In general, percent errors less than

### Percent Error Summaries (implemented in main_functions.py as `summarize_percent_error`)

Once the error signals are computed, we can compute statistics to characterize the error.  More importantly, we can compute the amount of time the error signal(s) exceed a particular threshold.  I recommend using the percent error signals for this summarization for the reasons described above, thus the threshold specified is a percent error threshold.

Currently, I compute the following threshold-based summaries:

1. total amount of time the percent error exceeds a threshold
2. fraction of time (with respect to the total activity time) that the percent error exceeds the threshold

I also compute the following statistics:

1. mean percent error
2. standard deviation of percent error
3. median percent error
4. max percent error
5. min percent error

## Rolling Average Threshold Error (implemented in main_functions.py as `compute_RAT_error`)

The error signals may be a little too fine-grained and noisy to compute useful error flags (i.e. booleans such that 'True' evaluations signifies some significant error that requires manual review). For example, I have not come across a single dataset where the percent error is below 0.5 for the entire duration of the dataset.
For a given RAT_window, computes the fraction of time the error signal within the window is greater than some error threshold.

### RAT Error Flag

Currently only one automatic error flag is built in to detect time-periods where the percent error exceeds. Essentially this detects if there is one or more periods of time where there are a large number of errors.  This is implemented in `compute_RAT_error` in main_functions.py.

### Suggestions for error detection

1. In the short-term, I recommend building automatic error detection algorithms to target known error types. For example, we can roughly estimate the fraction of time the Live algorithm falsely detects breaths by the Live-to-PP VT percent error signal.  By computing the fraction of time this error signal exceeds a reasonable threshold (e.g. 0.5), we can characterize the fraction of false-positive breaths detected by the Live algorithm.  Furthermore, an automatic flag for this type of error could be if this fraction exceeded some threshold (e.g. 0.1% of the time).
2. Any thresholds will have to be manually tuned and may differ between metrics.

## Other notes

1. For any error analysis,the first 60-120 seconds should be omitted because each algorithm needs to detect a certain number of breaths to properly calibrate
