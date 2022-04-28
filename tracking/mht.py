import sys, os
sys.path.insert(1, os.getcwd())


import numpy as np
import importlib
import tracking as tr
from itertools import product
import matplotlib.pyplot as plt


# %% Function definitions
def init_gate(q1, q2, dt, vmax=12000.0):
    if np.linalg.norm(q1 - q2) <= vmax * dt:
        return True
    else:
        return False


def update_inf(individual_hyp):
    for i in range(len(np.where(np.array(individual_hyp) == -np.inf)[0])):
        pass


def pair_hyp(hyp):
    pass


def eval_hyp(hyp):
    pass


def create_hyp_table(S0, S1, tracks, initial_hyp=False):
    # get numbers for new track hypothsis
    current_tracks = np.sort(list(tracks.keys()))
    new_track_start = current_tracks[-1]+1
    track_numbers = np.arange(new_track_start, new_track_start+len(S1))

    # create initial hypothesis table
    hyp_table = []
    for i in range(len(S1)):
        mn_hyp = [0]
        for j in range(len(S0)):
            if init_gate(S0[i], S1[j], 1, 0.75):
                mn_hyp.append(j + 1)
        mn_hyp.append(track_numbers[i])
        hyp_table.append(mn_hyp)

    # create all possible combinations
    combinations = [p for p in product(*hyp_table)]
    perm_table = np.asarray(combinations).T

    # remove impossible combinations
    non_zero_duplicates = []
    for i in range(len(perm_table[0])):
        # if there is a duplicate in column i+1 of perm_table, the value is saved in dup
        u, c = np.unique(perm_table[:, i], return_counts=True)
        dup = u[c > 1]

        # if there are non-zero duplicates, non_zero_duplicates gets a True, otherwise it gets a false
        non_zero_duplicates.append(np.any(dup > 0))

    hyp_possible = np.delete(perm_table, non_zero_duplicates, axis=1)

    return hyp_possible


# %% Data import and transformation
imports = ["snr50/truth1.txt", "snr50/truth2.txt", "nfft_15k/false.txt"]

# Slice data
_data = []
for i, file_ in enumerate(imports):
    _data.append(np.array(tr.import_data(file_)).T)

    if i == 0:
        _data[0][:, 0] = np.array(_data[0][:, 0]) - 5
    if i == 2:
        _data[2][:, 0] = np.array(_data[2][:, 0]) + 5

data_ = np.concatenate((_data[0],_data[1]))
data_ = np.concatenate((data_,_data[2]))
data = data_[data_[:,0].argsort()]
data = data[:12]

#Convert to cartesian coordinates
time_xyz = tr.conversion(data)
timesort_xyz = tr.time_slice(time_xyz)

# %% Initial gating
print("done")
