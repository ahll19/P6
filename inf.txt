import sys
import os
import numpy as np
from itertools import product
from scipy import special
import matplotlib.pyplot as plt

sys.path.insert(1, os.getcwd())
import tracking as tr

# Global variables ------------------------------------------------------------
# how many tracks have been created
total_tracks = 0

# wiki "standard gravitational parameter"
mu = 3.986004418e14

# Used in calculating hyp proba.
snr = 50
P_FA = np.exp(-10)
P_D = 0.5 * special.erfc(special.erfcinv(2 * P_FA) - np.sqrt(snr / 2))


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
    dm1 = (m1[1:] - m0[1:]) / dt
    x = np.hstack((m1[1:], dm1))
    x_predict = phi @ x
    m_predict = phi @ np.eye(6) @ phi.T + np.eye(6)

    return x_predict, m_predict


def __N_pdf(mean, Sig_inv):
    Sig = np.linalg.inv(Sig_inv)
    n = len(Sig)
    part1 = np.exp(-0.5 * mean.T @ Sig_inv @ mean)
    part2 = np.sqrt((2 * np.pi) ** n * np.abs(np.linalg.det(Sig)))

    return 20#(part1) / (part2)


def __beta_density(NFFT, n):
    n_FA = 1#NFFT * np.exp(-10)
    beta_FT = n_FA / n
    beta_NT = (n - n_FA) / n
    return beta_FT, beta_NT


def __Pik(H, P_g=1, P_D=0.2, NFFT=15000, kal_info=None):
    """
    y_t is measurements at time t, where y_t_hat is a prediction at time t-1.
    The calculation y_t[i]-y_t_hat[i] corresponds to the calculation in the
    kalman gate. They are both 2d arrays with the same amount of rows and 3
    columns. Sig_inv is the same cov matrix as the one from the gate and is a
    3d array, e.g., np.array([np.eye(3),np.eye(3)]) in the case with 2 predic-
    tions.

    Parameters
    ----------
    H : Hypothesis matrix. len(columns)=amount of hypothesis, len(rows)= amount
    of points
    c : Scaling of probability. The default is 1.
    P_g : Probability relating to prior points (only used if prior_info=True).
    The default is 0.2.
    P_D : Probability for detection. The default is 0.2.
    prior_info : Boolian - if True then we have prior info, Talse otherwise.
    The default is False.
    y_t : Measurements at time t. The default is [].
    y_t_hat : Predictions at time t. The default is [].
    Sig_inv : Covariance matrix from Kalman. The default is [].

    Returns
    -------
    prob : Array type of probabilities for each hypothesis

    """

    beta_FT, beta_NT = __beta_density(NFFT, len(H))
    N_TGT = np.max(H) - len(H)  # Number of previously known targets

    prob = np.zeros(len(H[0]))
    for i, hyp in enumerate(H.T):
        N_FT = np.count_nonzero(hyp == 0)  # Number of false
        N_NT = np.count_nonzero(hyp >= N_TGT + 1)  # Number of known targets in hyp
        N_DT = len(hyp) - N_FT - N_NT  # Number of prioer targets in given hyp

        prob[i] = (P_D ** N_DT * (1 - P_D) ** (N_TGT - N_DT) * beta_FT ** N_FT * beta_NT ** N_NT)
        prob[i] += 1e-10
        if kal_info is not None and N_DT >= 1:
            product = 1
            y_t, y_hat_t, Sig_inv = 0,0,0
            for j,item in enumerate(hyp):
                b = [(k, kal_info.index((j,item))) for k, kal_info in enumerate(kal_info) if (j,item) in kal_info]
                if len(b) == 0:
                    continue
                b = b[0][0]
                
                y_t, y_hat_t, Sig_inv = kal_info[b][1:]

                product *= __N_pdf(y_t - y_hat_t, Sig_inv)
                #print(product,prob[i])

            prob[i] *= product * P_g
            

    prob_hyp = np.vstack((prob, H))

    prob_hyp = prob_hyp.T[prob_hyp.T[:, 0].argsort()[::-1]].T
    
    prob_hyp[0] = prob_hyp[0]/sum(prob_hyp[0])
    #print(prob_hyp[0])
    return prob_hyp


def __prune(prob_hyp, th=0.1, N_h=5):
    if len(np.where(prob_hyp[0] < th)[0]) > 0:
        cut_index = np.min(np.where(prob_hyp[0] < th))
    
        pruned_hyp = prob_hyp[:, :cut_index]
    
        if len(pruned_hyp[0]) >= N_h:
            pruned_hyp = pruned_hyp[:, :N_h]
    
        return pruned_hyp
    else:
        if len(prob_hyp[0]) >= N_h:
            prob_hyp = prob_hyp[:, :N_h]
        return prob_hyp


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

    Det er ikke en rigtig mahalanobis norm, da vores estimat af variancen er
    litteral trash

    """
    # Keep track (haha) of the numbering of new tracks

    global total_tracks
    ls = len(s1)
    new_track_numbers = np.arange(total_tracks, total_tracks + ls)
    total_tracks += ls

    new_table = []
    kalman_info = []
    kalman_info_all = []
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
                    dt = np.abs(m1[0] - s0[row][0])
                    vmax = args[0]
                    d_dist = m1[1:] - s0[row][1:]
                    if np.linalg.norm(d_dist) <= vmax * dt:
                        mn_hyp.append(hyp_table[0, row, col])

                else:
                    # Mahalanobis gating (not really though), see docstring.
                    # predictions must be a dictionary with the keys
                    # representing the tracks (x, m)
                    track_num = hyp_table[0, row, col]
                    x = predictions[track_num][0][0][:3]
                    
                    
                    m = predictions[track_num][0][1][:3, :3]
                    d = (x - m1[1:]).T @ np.linalg.inv(m) @ (x - m1[1:])
                    threshold = args[1]
                    
                    
                    if d < threshold:
                        mn_hyp.append(hyp_table[0, row, col])
                        if [i,int(hyp_table[0, row, col])] not in kalman_info:
                            kalman_info.append([i,int(hyp_table[0, row, col])])
                            kalman_info_all.append([(i,int(hyp_table[0, row, col])),m1[1:],x,np.linalg.inv(m)])

        # save the potential of new track in the hypothesis
        mn_hyp.append(new_track_numbers[i])
        new_table.append(mn_hyp)

    new_tab_remove_doubles = []
    for i in range(len(new_table)):
        new_tab_remove_doubles.append(list(set(new_table[i])))
    
    new_table = new_tab_remove_doubles
    
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
        

    if len(kalman_info_all) > 0:
        probability_hyp_table = __Pik(hyp_possible, kal_info=kalman_info_all, P_D=P_D)
    else:
        probability_hyp_table = __Pik(hyp_possible, P_D=P_D)
    
    # Prune the hypothesis table
    _pruned_table = __prune(probability_hyp_table)
    pruned_probabilities = _pruned_table[0]
    pruned_table = _pruned_table[1:]

    # add an array which indicates which tracks are new and which are old
    new_track_indc = np.zeros_like(pruned_table)
    predictions1 = dict()

    for col in range(len(pruned_table[0, :])):
        for row in range(len(pruned_table[:, 0])):
            # check if the track is not a new one
            # and if the track is not 0
            condition_1 = pruned_table[row, col] not in new_track_numbers
            condition_2 = pruned_table[row, col] != 0
            if condition_1 and condition_2:
                # indicate that this entry is part of an old track
                new_track_indc[row, col] = 1
                track_num = pruned_table[row, col]
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
                idx_track = np.where(hyp_table[0] == track_num)[0]
                old_point = s0[idx_track[0]]

                prediction = __predict(old_point, s1[row])
                predictions1[track_num] = (prediction, row)

    final_table = np.stack((pruned_table, new_track_indc))

    return s1, final_table, predictions1


# Data import -----------------------------------------------------------------
# import data
imports = ["snr50/truth1.txt", "snr50/truth2.txt", "nfft_15k/false.txt"]

_data = []
for i, file_ in enumerate(imports):
    _data.append(np.array(tr.import_data(file_)).T)


data_ = np.concatenate((_data[0], _data[1]))
data_ = np.concatenate((data_, _data[2]))
data = data_[data_[:, 0].argsort()]
data = data[:]

time_xyz = tr.conversion(data)
timesort_xyz = tr.time_slice(time_xyz)  # point sorted by time [t, x, y, z]

# Testing the code ------------------------------------------------------------
# initial points n shit
old_points = timesort_xyz.pop(0)
old_hyp, s1 = init_tracking(old_points)
new_predicts = dict()

# saving the results
results = []
new_points = timesort_xyz[0]

track1 = np.zeros((len(timesort_xyz)-1,3))
track2 = np.zeros((len(timesort_xyz)-1,3))
for i in range(len(timesort_xyz)):
    new_points = timesort_xyz[i]
    iter_results = iter_tracking(
        old_points, new_points, old_hyp, new_predicts, args=(12000, 10e6))
    old_points = iter_results[0]
    old_hyp = iter_results[1]
    new_predicts = iter_results[2]
    # print(new_predicts)

    results.append(iter_results)
    most_likely_hyp = iter_results[1][0][:, 0]
    for k, t in enumerate(most_likely_hyp):
        if t == 1:
            track1[i] += timesort_xyz[i][int(k)][1:]
        if t == 2:
            track2[i] += timesort_xyz[i][int(k)][1:]

'''
for i in range(len(timesort_xyz)):
    print(i)
    new_points = timesort_xyz[i]
    iter_results = iter_tracking(old_points, new_points, old_hyp, new_predicts, args=(12000, 10e6))
    old_points = iter_results[0]
    old_hyp = iter_results[1]
    new_predicts = iter_results[2]
    #print(new_predicts)

    results.append(iter_results)
'''
print("==========================================")
print("Tables:")
for i in range(len(results)):
    print(results[i][1][0])
print("==========================================")

plt.scatter(track1[:,1], track1[:,2])
plt.scatter(track2[:,1], track2[:,2])
