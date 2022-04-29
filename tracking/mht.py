import sys
import os

sys.path.insert(1, os.getcwd())

import numpy as np
import tracking as tr
from itertools import product


# %% Function definitions
def init_gate(q1, q2, dt, vmax=12000.0):
    if np.linalg.norm(q1 - q2) <= vmax * dt:
        return True
    else:
        return False


def mah_gate(prediction, observation, threshold):
    """
    Create Mahalanobis gate for a prediction and observation
    :param prediction: tuple (returned by make_prediction from KalmanMHT) of state and variance prediction
    :param observation: new observation (can be [t, x, y, z] or [x, y, z] array)
    :param threshold: threshhold distance thang
    :return: True if the point is in the gate, False otherwise
    """
    _x, m = prediction[0], prediction[1]
    y = observation

    x = _x[:3]
    if len(y) == 4:
        y = y[1:]

    d = (x-y).T@np.linalg.inv(m)@(x-y)
    if d < threshold:
        return True
    else:
        return False


def N_pdf(mean, Sig_inv):
    Sig = np.linalg.inv(Sig_inv)
    n = len(Sig)
    return (-0.5 * (mean.T @ Sig_inv @ mean)) / \
           (np.sqrt((2 * np.pi) ** n * np.linalg.norm(Sig)))


def Pik(H, c=1, P_g=0.2, P_D=0.2, prior_info=False, y_t=None, y_t_hat=None, Sig_inv=None):
    """
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

    N_TGT = np.max(H) - len(H)  # Number of previously known targets

    prob = np.zeros(len(H[0]))
    for i, hyp in enumerate(H.T):
        N_FT = np.count_nonzero(hyp == 0)  # Number of false
        N_NT = np.count_nonzero(hyp >= N_TGT + 1)  # Number of known targets in hyp
        N_DT = len(hyp) - N_FT - N_NT  # Number of prioer targets in given hyp

        beta_FT = N_FT / (N_NT + N_DT + N_FT)
        beta_NT = N_NT / (N_FT + N_DT + N_NT)

        prob[i] = (1 / c) * (P_D ** N_DT * (1 - P_D) ** (N_TGT - N_DT) * beta_FT ** N_FT * beta_NT ** N_NT)

        if prior_info:
            product = 1
            for j in range(N_DT):
                product *= N_pdf(y_t[j] - y_t_hat[j],
                                 Sig_inv)  # Må være den prediction der hører til givet punkt der menes

            prob[i] *= product * P_g

    prob_hyp = np.vstack((prob, H))

    prob_hyp = prob_hyp.T[prob_hyp.T[:, 0].argsort()[::-1]].T

    return prob_hyp


def prune(prob_hyp, th=0.1, N_h=10000):
    cut_index = np.min(np.where(prob_hyp[0] < th))

    pruned_hyp = prob_hyp[:, :cut_index]

    if len(pruned_hyp[0]) >= N_h:
        pruned_hyp = pruned_hyp[:, :N_h]

    return pruned_hyp


def create_init_hyp_table(S0, S1, tracks):
    """
    NOTE: rewrite such that S0 is not needed (can be gained from tracks)
    Given a set of measurements, uses gating to list all possible new hypothesis
    :param S0: Last points from each track (e.g. [track1[-1], track2[-1], ...])
    :param S1: New measurements
    :param tracks:
    :param initial_hyp:
    :return:
    """
    # get numbers for new track hypothsis
    current_tracks = np.sort(list(tracks.keys()))
    new_track_start = current_tracks[-1] + 1
    track_numbers = np.arange(new_track_start, new_track_start + len(S1))

    # create initial hypothesis table
    hyp_table = []
    for i in range(len(S1)):
        mn_hyp = [0]
        for j in range(len(S0)):
            if init_gate(S0[i], S1[j], 1):
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


def create_hyp_table(S0, S1, tracks):
    # tbd: lav en funktion som bruger den nye gate til at lave en hyp_table

    return None


def assign_hyp_to_tracks(tracks, hyp_table):
    """
    Function which takes the current tracks, and a hypothesis table,
    and returns possible track assignments for each new data point
    :param tracks: dictionary of all current tracks
    :param hyp_table: hypothesis table (returned by create_hyp_table
    :return: Returns a list of sets. Each index of the list corresponds to one of the new
             data points. The set at each index indicates which track could be assigned to the new
             data point.
    """
    # get all track keys, without 0
    all_tracks = tracks.copy()
    del all_tracks[0]
    all_track_keys = list(all_tracks.keys())
    num_points = hyp_table.shape[0]

    # create a list where the first index saves possible tracks for the first point etc.
    point_possible_tracks = [set() for i in range(num_points)]
    for hyp in hyp_table.T:
        for k in all_track_keys:
            # check if a hypothesis implies a point belongs to a current track
            _key_check = np.where(hyp == k)[0]
            is_not_empty = _key_check.size != 0

            if is_not_empty:
                # add points to the tracks the hypothesis specifies
                point_possible_tracks[_key_check[0]].add(hyp[_key_check[0]])

    return point_possible_tracks


def kalman_step(tracks, kalman):
    for k in kalman:
        kalman[k].make_prediction()
        print(kalman[k].x_predictions)


def get_state_in_track(track, idx=None):
    """
    NOTE: use velocity_algo_pair, it's better (and more work)
    Gets the state from a track. If idx is not given the functions uses
    the lates points in track to generate a new state.
    :param track: The track to get the state from
    :param idx: Tuple containing 2 indeces (prev point, next point).
    :return: returns a state vector [x, y, z, xdot, ydot, zdot]
    """
    if idx is None:
        x1, x2 = track[-2], track[-1]
        dt = x2[0]-x1[0]
        dx = (x2[1:]-x1[1:])/dt
    else:
        x1, x2 = track[idx[1]], track[idx[0]]
        dt = x2[0]-x1[0]
        dx = (x2[1:]-x1[1:])/dt

    state = np.hstack((x2[1:], dx))

    return state


# %% Data import and transformation
imports = ["snr50/truth1.txt", "snr50/truth2.txt", "nfft_15k/false.txt"]

# slice data
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

# Convert to cartesian coordinates
time_xyz = tr.conversion(data)
timesort_xyz = tr.time_slice(time_xyz) # point sorted by time [t, x, y, z]

# %% tracking test
# create initial track
initial_track_keys = list(range(1, timesort_xyz[0].shape[1] + 1))
track_all_points = {0: []} # saving all tracks in a dict
for i in range(len(initial_track_keys)):
    _key = initial_track_keys[i]
    _point = timesort_xyz[0][0, i]
    track_all_points[_key] = [_point]

# assume [{1}, {2}, set()] from assign_hyp_to_track is true
track_all_points[1].append(timesort_xyz[1][0, 0])
track_all_points[2].append(timesort_xyz[1][0, 1])

# create a dictionary with states in tracks (contains one less point than the tracks dict)
track_states = dict()
state1 = get_state_in_track(track_all_points[1])
state2 = get_state_in_track(track_all_points[2])

# append states
track_states[1] = [state1]
track_states[2] = [state2]

# %% try out kalman tracks_test_state
# start kalman filters
s_u, s_w, m_init = [np.eye(6)]*3
track_filters = dict()

# initialize kalman filters
for k in track_states:
    track_filters[k] = tr.KalmanMHT(s_u, s_w, m_init, track_states[k][0])

# create gate for each prediction


# %% testing
hyp1_table = create_init_hyp_table(timesort_xyz[0][0, :, 1:], timesort_xyz[1][0, :, 1:], track_all_points)
hyp_prob = Pik(hyp1_table)
assign_hyp_to_tracks(track_all_points, hyp1_table)
