import sys
import os
import numpy as np
from itertools import product

sys.path.insert(1, os.getcwd())
import tracking as tr


# Global variables ------------------------------------------------------------
total_tracks = 0
mu = 3.986004418e14  # wiki "standard gravitational parameter"


# Intermediate functions ------------------------------------------------------
def __predict(m0, m1):
    global mu
    r_i, r_j, r_k = m1[1], m1[2], m1[3]
    r = np.sqrt(m1[1] ** 2 + m1[2] ** 2 + m1[3] ** 2)
    F1, F2, F4 = np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))
    F3 = np.asanyarray([[-mu / (r ** 3) + (3 * mu * r_i ** 2) / (r ** 5),
                         (3 * mu * r_i * r_j) / (r ** 5), (3 * mu * r_i * r_k) / (r ** 5)],
                        [(3 * mu * r_i * r_j) / (r ** 5),
                         -mu / r ** 3 + (3 * mu * r_j ** 2) / (r ** 5),
                         (3 * mu * r_j * r_k) / (r ** 5)],
                        [(3 * mu * r_i * r_k) / (r ** 5), (3 * mu * r_j * r_k) / (r ** 5),
                         -mu / (r ** 3) + (3 * mu * r_k ** 2) / (r ** 5)]])
    F_top = np.concatenate((F1, F2), axis=1)
    F_bot = np.concatenate((F3, F4), axis=1)
    F = np.concatenate((F_top, F_bot))

    dt = np.abs(m1[0] - m0[0])
    phi = np.eye(6) + F * dt

    # m_predict is a static size, and does not change with different states
    # this is a simplification, and is subject to change if the algorithm
    # does not work
    dm1 = (m1[1:] - m0[1:])/dt
    x = np.hstack((m1[1:], dm1))
    x_predict = phi @ x
    m_predict = phi @ np.eye(6) @ phi.T + np.eye(6)

    return x_predict, m_predict


# Main Functions --------------------------------------------------------------
def init_tracking(init_point):
    # create a the initial hypothesis table
    global total_tracks
    ls = len(init_point)
    total_tracks += ls

    hyp_table = np.zeros((2, ls, 2))
    for i in range(ls):
        hyp_table[0, i, 1] = i + 1

    return hyp_table, init_point


def iter_tracking(s0, s1, hyp_table, predictions, args=None):
    """
    Do some cool stuff

    args = (vmax, threshold)
    vmax = max velocity of satellite, used for simple gating
    threshold = threshold for the "mahalanobis" norm

    """
    # Keep track (haha) of the numbering of new tracks

    global total_tracks
    ls = len(s1)
    new_track_numbers = np.arange(total_tracks, total_tracks + ls)
    total_tracks += ls

    new_table = []
    for i, m1 in enumerate(s1):
        mn_hyp = [0]
        for col in range(len(hyp_table[1, 0, :])):
            for row in range(len(hyp_table[1, :, 0])):
                # Don't gate on "empty" tracks
                if hyp_table[0, row, col] == 0:
                    continue

                # Check which gate to use
                new_track = not bool(hyp_table[1, row, col])
                if new_track:
                    # Simple gate for new tracks
                    dt = np.abs(m1[0]-s0[row][0])
                    vmax = args[0]
                    d_dist = m1[1:] - s0[row][1:]
                    if np.linalg.norm(d_dist) <= vmax * dt:
                        mn_hyp.append(hyp_table[0, row, col])
                else:
                    # Mahalanobis gating (not really though)
                    # predictions must be tupe of 2 lists (x, m)
                    x, m = predictions[0][row], predictions[1][row]
                    d = (x - m1[1:]).T @ np.linalg.inv(m) @ (x - m1[1:])
                    threshold = args[1]
                    if d < threshold:
                        mn_hyp.append(hyp_table[0, row, col])

        # save the potential of new track in the hypothesis
        mn_hyp.append(new_track_numbers[i])
        new_table.append(mn_hyp)

    # permutate the hypothesis table
    combinations = [p for p in product(*new_table)]
    perm_table = np.asarray(combinations).T

    # remove impossible combinations
    non_zero_duplicates = []
    for i in range(len(perm_table[0])):
        # if there is a duplicate in column i+1 of perm_table, the value is saved in dup
        u, c = np.unique(perm_table[:, i], return_counts=True)
        dup = u[c > 1]

        # if there are non-zero duplicates, non_zero_duplicates gets a True, otherwise it gets a false
        non_zero_duplicates.append(np.any(dup > 0))

    # create array of possible hypotheses
    hyp_possible = np.delete(perm_table, non_zero_duplicates, axis=1)

    # add an array which indicates which tracks are new and which are old
    new_track_indc = np.zeros_like(hyp_possible)
    predictions = []

    for col in range(len(hyp_possible[0, :])):
        for row in range(len(hyp_possible[:, 0])):
            # check if the track is not a new one
            # and if the track is not 0
            condition_1 = hyp_possible[row, col] not in new_track_numbers
            condition_2 = hyp_possible[row, col] != 0
            if condition_1 and condition_2:
                # indicate that this entry is part of an old track
                new_track_indc[row, col] = 1
                track_num = hyp_possible[row, col]
                # create a prediction for this entry
                """
                NOTE:
                as of now, the algorithm assumes that each track will only
                corrospond to one previous point. Meaning that each row of
                a hypothesis table will only have a maximum of one 
                instance of a track. If more instances occur the algorithm
                will not handle the multiple predictions necessary to
                continue with the gating of the next points.
                Only the first instance where the track is found in the
                old track is use to create predictions
                """
                idx_track = np.where(hyp_table == track_num)[0]
                old_point = s0[idx_track[0]]

                prediction = __predict(old_point, s1[row])
                predictions.append(prediction)
                """
                TODO:
                find out how to save predictions, such that the gating can be
                done on the right points.
                """

    return predictions



# Data import -----------------------------------------------------------------
# import data
imports = ["snr50/truth1.txt", "snr50/truth2.txt", "nfft_15k/false.txt"]

_data = []
for i, file_ in enumerate(imports):
    _data.append(np.array(tr.import_data(file_)).T)

    if i == 0:
        _data[0][:, 0] = np.array(_data[0][:, 0]) - 5
    if i == 2:
        _data[2][:, 0] = np.array(_data[2][:, 0]) + 5

data_ = np.concatenate((_data[0], _data[1]))
data_ = np.concatenate((data_, _data[2]))
data = data_[data_[:, 0].argsort()]
data = data[:12]

time_xyz = tr.conversion(data)
timesort_xyz = tr.time_slice(time_xyz)  # point sorted by time [t, x, y, z]


# Testing the code ------------------------------------------------------------
points0, points1 = timesort_xyz[0], timesort_xyz[1]
hyp1, p1 = init_tracking(points0)
_tester = iter_tracking(points0, points1, hyp1, [0], args=(12000, 10e4))
