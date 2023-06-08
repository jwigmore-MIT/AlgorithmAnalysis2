import numpy as np
from copy import deepcopy


def distance(r: tuple,q: tuple):
    return abs(r[1]-q[1])

def shift_error(r: tuple,q: tuple, avg_shift = 0, gamma = 2):
    return abs(r[0] - q[0] - avg_shift)**gamma

def dtw_distance(qry, ref, window, avg_shift = False, gamma = 2, beta = 0.4):
    '''
    Rules:
        1. each element of query can only be matched to a preceding element of reference
        2. each element is only matched to one query
        3. each element of query can only be matched to an element of reference that is within the window
        4. if there does not exist an element of reference within the window, then the element of query is matched to the first element outside of the window
    :param ref: (x,y) pairs of the reference signal
    :param qry: (x,y) pairs of the query signal
    :param window: the maximum x distance between the reference and query signal
    :return:
    '''
    # Initialize
    n = len(qry)
    m = len(ref)
    DTW = np.zeros((n,m))
    DTW[0,0] = 0
    i_to_j = np.zeros(n, dtype=int)# a map from each element in the query to an element in the reference
    shift_avg = window[1]/2
    shift_record = []
    # Loop through each element of the query
    for i,q in zip(range(n),qry):
        # i is the index of the query
        # q is the element of the query
        # Loop through each element of the reference
        min_r0 = q[0] - window[1] # Minimum x value of reference that can be matched to query
        max_r0 = q[0] - window[0]
        min_cost = np.inf
        # get all elements of reference that can be matched to query
        ref_window = ref[(ref[:,0] >= min_r0) & (ref[:,0] <= max_r0)]
        # get index of each element in ref_window
        ref_window_index = np.where((ref[:,0] >= min_r0) & (ref[:,0] <= max_r0))[0]
        #print(f"q: {q}, i, :{i},  ref_window: {ref_window}, valid_j's: {ref_window_index}")
        for j,r in zip(ref_window_index,ref_window):
            if avg_shift:
                cost = distance(r,q) + shift_error(r,q,shift_avg, gamma)
            else:
                cost = distance(r,q)
            if cost <= min_cost:
                min_cost = cost
                i_to_j[i] = j

            DTW[i,j] = cost + min(DTW[i-1,j-1],DTW[i-1,j],DTW[i,j-1])
        if i_to_j[i] == 0:
            i_to_j[i] = i_to_j[i-1]
        if avg_shift:
            # get exponential average of the shift
            if i == 0:
                shift_avg = shift_avg
            else:
                shift_avg = (1-beta)*shift_avg + beta*(q[0] - ref[i_to_j[i]][0])
            shift_record.append((q[0] - ref[i_to_j[i]][0],shift_avg))
    if avg_shift:
        return DTW, i_to_j, shift_record
    else:
        return DTW, i_to_j

def dtw2(qry, ref, window):
    '''
    Perform dtw without any contraints on direction within a fixed window
    :param qry:
    :param ref:
    :param window:
    :return:
    '''
    n = len(qry)
    m = len(ref)
    DTW = np.ones((n, m))*np.inf
    DTW[0, 0] = 0
    i_to_j = np.zeros(n, dtype=int)  # a map from each element in the query to an element in the reference
    # Loop through each element of the query
    for i, q in zip(range(n), qry):
        # i is the index of the query
        # q is the element of the query
        # Loop through each element of the reference
        min_r0 = q[0] - window
        max_r0 = q[0] + window
        min_cost = np.inf
        # get all elements of reference that can be matched to query
        ref_window = ref[(ref[:, 0] >= min_r0) & (ref[:, 0] <= max_r0)]
        # get index of each element in ref_window
        ref_window_index = np.where((ref[:, 0] >= min_r0) & (ref[:, 0] <= max_r0))[0]
        # print(f"q: {q}, i, :{i},  ref_window: {ref_window}, valid_j's: {ref_window_index}")
        for j, r in zip(ref_window_index, ref_window):
            cost = distance(r, q)

            DTW[i, j] = cost + min(DTW[i - 1, j - 1], DTW[i - 1, j], DTW[i, j - 1])
        if i_to_j[i] == 0:
            i_to_j[i] = i_to_j[i - 1]
    return DTW


def get_warp_path(DTW):
    '''
    Warp the query signal to the reference signal by finding the minimum cost path through the DTW matrix

    :param qry:
    :param ref:
    :param DTW:
    :return:
    '''
    # Initialize
    N = DTW.shape[0]
    M = DTW.shape[1]
    path = []
    n = N-1
    m = M-1
    while n > 0 and m > 0:
        if n == 0:
            cell = (n,m-1)
        if m == 0:
            cell = (n-1,m)
        else:
            val = min(DTW[n-1,m-1],DTW[n-1,m],DTW[n,m-1])
            if val == DTW[n-1,m-1]:
                cell = (n-1,m-1)
            elif val == DTW[n-1,m]:
                cell = (n-1,m)
            else:
                cell = (n,m-1)
        path.append(cell)
        (n,m) = cell

    path.reverse()
    return path

def remap_query(qry, ref, i_to_j):
    '''
    Remap the query signal to the reference signal by using the i_to_j map
    :param qry:
    :param ref:
    :param i_to_j:
    :return:
    '''
    # Initialize
    n = len(qry)
    m = len(ref)
    remapped_qry = np.zeros((n,2))
    for i in range(n):
        remapped_qry[i,0] = ref[int(i_to_j[i]),0]
        remapped_qry[i,1] = qry[i,1]

    return remapped_qry

def fix_i_to_j(i_to_j_og, qry):
    # in reverse order of i_to_j, if
    i_to_j = deepcopy(i_to_j_og)
    for i in range(len(i_to_j)-1,1,-1):
        qry_i = qry[i]
        qry_i_1 = qry[i-1]
        if i_to_j[i] == i_to_j[i-1] and qry_i[1] != qry_i_1[1]:
            i_to_j[i-1] -= 1
    return i_to_j