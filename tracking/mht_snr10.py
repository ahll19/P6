import sys
import os
import numpy as np
from itertools import product
from scipy import special
import matplotlib.pyplot as plt
import pywt
sys.path.insert(1, os.getcwd())
import tracking as tr
from kalman_help_me import KalmanGating

# Global variables ------------------------------------------------------------
# how many tracks have been created
total_tracks = 0

# wiki "standard gravitational parameter"
mu = 3.986004418e14
testnr = 3
wavelet = True
# Used in calculating hyp proba.
snr = 10
P_FA = np.exp(-10)
P_D = 0.5 * special.erfc(special.erfcinv(2 * P_FA) - np.sqrt(snr / 2))
NFFT = 15000
yo = []
all_hyp = [0]
sat_nr = 2

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
   # dm1 = (m1[1:] - m0[1:]) / dt
    #x = np.hstack((m1[1:], dm1))
    x = m1[1:]
    x_predict = phi @ x
    m_predict = phi @ np.eye(6) @ phi.T + np.eye(6)

    return x_predict, m_predict


def __N_pdf(mean, Sig_inv):
    Sig = np.linalg.inv(Sig_inv)
    n = len(Sig)
    mean *= 1/1000000
    part1 = (-0.5 * mean.T @ Sig_inv @ mean)
    part2 = np.log(np.sqrt((2 * np.pi) ** n * np.abs(np.linalg.det(Sig))))
    return (part1) - (part2)


def __beta_density(NFFT, N_NT):
    n_FA = int(NFFT * np.exp(-10))
    beta_FT = n_FA / (4.54*10**16) + 1/10**20
    beta_NT = (N_NT) / (4.54*10**16) + 1/10**20
    return beta_FT, beta_NT


def __Pik(H, P_g=1, P_D=0.2, kal_info=None, N_TGT = 2, old_probs = None):
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
    P_g : Probability relating to prior points (only used if prior_info=True).
    The default is 0.2.
    P_D : Probability for detection. The default is 0.2.
    NFFT : An input from matlab code used in the normal distribution pdf.
    kal_info : Consists of a list with a tuple 

    Returns
    -------
    prob : Array type of probabilities for each hypothesis

    """
    global NFFT, yo
    #NFFT = 15000
    beta_FT, beta_NT = __beta_density(NFFT, len(H))
    if type(old_probs) != int :
        norm_old_hyp = old_probs[0]
        norm_old = (norm_old_hyp - norm_old_hyp.min())/ (norm_old_hyp.max() - norm_old_hyp.min())
        old_probs[0] = norm_old * np.max(old_probs[0])
    prob = np.zeros(len(H[0]))
    for i, hyp in enumerate(H.T):
        N_FT = np.count_nonzero(hyp == 0)  # Number of false
        N_NT = np.count_nonzero(hyp >= np.max(H) - len(H) + 1)  # Number of known targets in hyp
        N_DT = len(hyp) - N_FT - N_NT  # Number of prioer targets in given hyp
        #prob[i] = np.log((P_D ** N_DT * (1 - P_D) ** (N_TGT - N_DT) * beta_FT ** N_FT * beta_NT ** N_NT))
        prob[i] = np.log(P_D)*N_DT + np.log(1 - P_D)*(N_TGT - N_DT) + np.log( beta_FT)* N_FT + np.log(beta_NT) * N_NT
        P_g = 0
        if kal_info is not None and N_DT >= 1:
            product = 0
            y_t, y_hat_t, Sig_inv = 0,0,0
            for j,item in enumerate(hyp):
                b = [(k, kal_info.index((j,item))) for k, kal_info in enumerate(kal_info) if (j,item) in kal_info]
                yo.append(kal_info)
                if len(b) == 0:
                    continue
                b = b[0][0]
                
                y_t, y_hat_t, Sig_inv = kal_info[b][1:]

                product += __N_pdf(y_t - y_hat_t, Sig_inv)
                P_g_index = np.where(old_probs == item)[1]
                P_g += np.sum(old_probs[0,P_g_index])
                
            
            prob[i] += product + 0 #P_g
    prob_hyp = np.vstack((prob, H))
    
    #prob_hyp[0] += np.max(prob_hyp[0])
    prob_hyp = prob_hyp.T[prob_hyp.T[:, 0].argsort()[::-1]].T
   

    return prob_hyp


def __prune(prob_hyp, th=0.1, N_h=1):
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


def iter_tracking(s0, s1, hyp_table, predictions, args=None, tracks=None, old_probs = None):
    """
    Do some cool stuff

    args = (vmax, threshold)
    vmax = max velocity of satellite, used for simple gating
    threshold = threshold for the "mahalanobis" norm

    Det er ikke en rigtig mahalanobis norm, da vores estimat af variancen er
    litteral trash

    """
    # Keep track (haha) of the numbering of new tracks

    global total_tracks, all_hyp
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
                    #print(np.linalg.norm(d_dist),dt)
                    if np.linalg.norm(d_dist) <= vmax * dt:
                        mn_hyp.append(hyp_table[0, row, col])
                        #print("hejhej")

                else:
                    # Mahalanobis gating (not really though), see docstring.
                    # predictions must be a dictionary with the keys
                    # representing the tracks (x, m)
                    track_num = hyp_table[0, row, col]
                    
                    x = predictions[track_num][0][0][:]
                    m = predictions[track_num][0][1][:, :]
                    d = (x - m1[1:]).T @ np.linalg.inv(m) @ (x - m1[1:])
                    threshold = args[1]
                    #print((x - m1[1:]),np.linalg.inv(m),d)
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
        probability_hyp_table = __Pik(hyp_possible, kal_info=kalman_info_all, P_D=P_D, N_TGT = np.count_nonzero(np.unique(hyp_table)), old_probs=old_probs)
    else:
        probability_hyp_table = __Pik(hyp_possible, P_D=P_D, N_TGT = np.count_nonzero(np.unique(hyp_table)), old_probs = old_probs)
    
    # Prune the hypothesis table
    _pruned_table = __prune(probability_hyp_table)
    all_hyp.append(_pruned_table)
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
                
                track_points = tracks[f'{int(track_num)}'].copy()

                track_points.append(s1[row])
                
                if False:#len(track_points)>2: #new kalman
                    # initialize kalman
                    init_x = track_points[0]
                    x_1 = track_points[1]
                    
                    mults = [0.1, 0.5, 0.01]
                    s_u, s_w, m_init = np.eye(6) * mults[0], np.eye(6) * mults[1], np.eye(6) * mults[2]
                    kalman = KalmanGating(s_u, s_w, init_x, m_init)
                    kalman.init_gate(x_1)
                    
                    # run kalman
                    for k in range(2, len(track_points)):
                        kalman.prediction(append_prediction=True)
                        kalman.observation(track_points[k])
                        
                    prediction = (kalman.state_predictions[-1],kalman.m_predictions[-1])
                    predictions1[track_num] = (prediction, row)
                else: 
                    #kalman = KalmanGating(s_u, s_w, old_point, m_init)
                    #kalman.init_gate(s1[row])
                    prediction = __predict(old_point,s1[row])
                    yo.append(prediction)
                    predictions1[track_num] = (prediction, row)
                    

    final_table = np.stack((pruned_table, new_track_indc))
    
    return s1, final_table, predictions1


# Data import -----------------------------------------------------------------
# import data
imports = ["snr"+str(snr)+"/truth1.txt", "snr"+str(snr)+"/truth2.txt", "snr"+str(snr)+"/truth3.txt", "snr"+str(snr)+"/truth4.txt", "snr"+str(snr)+"/truth5.txt", "nfft_"+str(NFFT)[:2]+"k/false.txt"]

_data = []
for i, file_ in enumerate(imports):
    _data.append(np.array(tr.import_data(file_)).T)
    
if testnr == 5:
    data_ = np.concatenate((_data[0], _data[1]))
    data_ = np.concatenate((data_, _data[2]))
    data_ = np.concatenate((data_, _data[3]))
    data_ = np.concatenate((data_, _data[4]))
    data_ = np.concatenate((data_, _data[5]))
    data = data_[data_[:, 0].argsort()]

if testnr == 2:
    data = _data[sat_nr-1] 
    
if testnr == 3:
    data_ = np.concatenate((_data[sat_nr-1], _data[5]))
    data = data_[data_[:, 0].argsort()]

data = data[:]
if testnr == 4:
    data = np.loadtxt("data4.txt") #For test 4
time_xyz = tr.conversion(data)
timesort_xyz = tr.time_slice(time_xyz) # point sorted by time [t, x, y, z]
# Testing the code ------------------------------------------------------------
# initial points n shit
old_points = timesort_xyz.pop(0)

old_vel = np.zeros((len(old_points),3))
old_points = np.hstack((old_points,old_vel))

old_hyp, s1 = init_tracking(old_points)
new_predicts = dict()

# saving the results
results = []
new_points = timesort_xyz[0]
#%%
tracks = {}
tracks_vel = dict()
tracks_weibel = {}
for i in range(0,len(time_xyz)+1):
    tracks[str(i)] = []


#%%
all_predicts = dict()
for i in range(len(timesort_xyz)):
    new_points = timesort_xyz[i]
    
    new_vel = np.zeros((len(new_points),3))
    new_points = np.hstack((new_points,new_vel))
    if i > 0:
        for ii in range(len(timesort_xyz[i])):
            for iii in range(len(timesort_xyz[i-1])):
                dist_travel = (timesort_xyz[i][ii][1:] - timesort_xyz[i-1][iii][1:])*0.1
                if np.linalg.norm(dist_travel) <= 10000:
                    new_points[ii,4:] += dist_travel
    iter_results = iter_tracking(
        old_points, new_points, old_hyp, new_predicts, args=(100000, 10e8),tracks=tracks, old_probs=all_hyp[i-1])
    old_points = iter_results[0]
    old_hyp = iter_results[1]
    new_predicts = iter_results[2]
    #print(old_hyp[0])
    #if i > 80:
    #    break
    for key, value in new_predicts.items():
        if key in all_predicts:
            if isinstance(all_predicts[key], list):
                all_predicts[key].append(np.concatenate([np.array([new_points[0][0]+0.1]),value[0][0][:3]]))
                tracks_vel[key].append(np.concatenate([np.array([new_points[0][0]+0.1]),value[0][0][3:]]))
            else:
                temp_list = [all_predicts[key]]
                temp_list.append(np.concatenate([np.array([new_points[0][0]+0.1]),value[0][0][:3]]))
                all_predicts[key] = temp_list
                temp_list1 = [tracks_vel[key]]
                temp_list1.append(np.concatenate([np.array([new_points[0][0]+0.1]),value[0][0][3:]]))
                tracks_vel[key] = temp_list1
                print("yo",i)
        else:
            all_predicts[key] = np.concatenate([np.array([new_points[0][0]+0.1]),value[0][0][:3]])
            tracks_vel[key] = (np.concatenate([np.array([new_points[0][0]+0.1]),value[0][0][3:]]))
    results.append(iter_results)
    most_likely_hyp = iter_results[1][0][:, 0]
    for k, t in enumerate(most_likely_hyp):
        tracks[str(int(t))].append(timesort_xyz[i][int(k)])
        if len(tracks[str(int(t))]) >=20 and str(int(t)) != "0" and wavelet:
            coeffs_x = pywt.wavedec(np.array(tracks[str(int(t))])[:,1], 'db4', level=pywt.dwt_max_level(len(tracks[str(int(t))]), 'db4')) 
            coeffs_y= pywt.wavedec(np.array(tracks[str(int(t))])[:,2], 'db4', level=pywt.dwt_max_level(len(tracks[str(int(t))]), 'db4')) 
            coeffs_z = pywt.wavedec(np.array(tracks[str(int(t))])[:,3], 'db4', level=pywt.dwt_max_level(len(tracks[str(int(t))]), 'db4')) 
            coeffs_x[-1][abs(coeffs_x[-1]) > 1] = 0
            coeffs_y[-1][abs(coeffs_y[-1]) > 1] = 0
            coeffs_z[-1][abs(coeffs_z[-1]) > 1] = 0
            x_coor = pywt.waverec(coeffs_x, 'db4')
            y_coor = pywt.waverec(coeffs_y, 'db4')
            z_coor = pywt.waverec(coeffs_z, 'db4')
            for j in range(0,len(tracks[str(int(t))])):
                tracks[str(int(t))][j][1] = x_coor[j]
                tracks[str(int(t))][j][2] = y_coor[j]
                tracks[str(int(t))][j][3] = z_coor[j]


#%% plot
entire_orb = []
entire_orb.append(tr.conversion(np.loadtxt("data4_sat1.txt")))
entire_orb.append(tr.conversion(np.loadtxt("data4_sat2.txt")))
entire_orb.append(tr.conversion(np.loadtxt("data4_sat3.txt")))
entire_orb.append(tr.conversion(np.loadtxt("data4_sat4.txt")))
entire_orb.append(tr.conversion(np.loadtxt("data4_sat5.txt")))

fig, axs = plt.subplots(3,1, sharex=True,sharey=False,figsize=(14,10))
fig.subplots_adjust(left=0.1, wspace=0.3)

# plotting data
if testnr == 5:
    fig.suptitle("Position, " + "SNR =" + str(snr) +", NFFT =" + str(NFFT)[:2]+"k",fontsize=29)

    alph = 0.7
    col = "k"
    axs[0].plot(entire_orb[0][:,0], entire_orb[0][:,1], color = col, linestyle='--', alpha = alph)
    axs[1].plot(entire_orb[0][:,0], entire_orb[0][:,2], label = "True orbits", color = col, linestyle='--', alpha = alph)
    axs[2].plot(entire_orb[0][:,0], entire_orb[0][:,3], color = col, linestyle='--', alpha = alph)
    for i in range(1,5):
        axs[0].plot(entire_orb[i][:,0], entire_orb[i][:,1], color = col, linestyle='--', alpha = alph)
        axs[1].plot(entire_orb[i][:,0], entire_orb[i][:,2], color = col, linestyle='--', alpha = alph)
        axs[2].plot(entire_orb[i][:,0], entire_orb[i][:,3], color = col, linestyle='--', alpha = alph)
if testnr == 3:
    fig.suptitle("Position, " + "SNR =" + str(snr) +", NFFT =" + str(NFFT)[:2]+"k",fontsize=29)

    alph = 0.7
    col = "k"
    axs[0].plot(entire_orb[sat_nr-1][:,0], entire_orb[sat_nr-1][:,1], color = col, linestyle='--', alpha = alph)
    axs[1].plot(entire_orb[sat_nr-1][:,0], entire_orb[sat_nr-1][:,2], label = "True orbits", color = col, linestyle='--', alpha = alph)
    axs[2].plot(entire_orb[sat_nr-1][:,0], entire_orb[sat_nr-1][:,3], color = col, linestyle='--', alpha = alph)

if testnr == 2:
    fig.suptitle(f"Satellite {sat_nr}",fontsize=29)

    alph = 1
    col = "r"
    axs[0].plot(entire_orb[sat_nr-1][:,0], entire_orb[sat_nr-1][:,1], color = col, linestyle='--', alpha = alph)
    axs[1].plot(entire_orb[sat_nr-1][:,0], entire_orb[sat_nr-1][:,2], label = "True orbits", color = col, linestyle='--', alpha = alph)
    axs[2].plot(entire_orb[sat_nr-1][:,0], entire_orb[sat_nr-1][:,3], color = col, linestyle='--', alpha = alph)


# plotting tracks
if testnr == 2:
    size = 10
    track_count = 1
    for i in range(1, len(time_xyz)):
        if len(tracks[str(i)]) >= 2:
            axs[0].scatter(np.array(tracks[str(i)])[:,0], np.array(tracks[str(i)])[:,1], s = size, label = "Kalman" , color="b")
            axs[0].grid(True)
            axs[0].set_ylabel("$r_y$ [m]")
            axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axs[1].scatter(np.array(tracks[str(i)])[:,0], np.array(tracks[str(i)])[:,2], s = size, label = "Kalman" , color="b")
            axs[1].grid(True)
            axs[1].set_ylabel("$r_y$ [m]")
            axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axs[2].scatter(np.array(tracks[str(i)])[:,0], np.array(tracks[str(i)])[:,3], s = size, label = "Kalman" , color="b")
            axs[2].grid(True)
            axs[2].set_xlabel("Time [s]")
            axs[2].set_ylabel("$r_z$ [m]")
            axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            track_count += 1
else:
    size = 10
    track_count = 1
    for i in range(1, len(time_xyz)):
        if len(tracks[str(i)]) >= 2:
            axs[0].scatter(np.array(tracks[str(i)])[:,0], np.array(tracks[str(i)])[:,1], s = size)
            axs[0].grid(True)
            axs[0].set_ylabel("$r_y$ [m]")
            axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axs[1].scatter(np.array(tracks[str(i)])[:,0], np.array(tracks[str(i)])[:,2], s = size)
            axs[1].grid(True)
            axs[1].set_ylabel("$r_y$ [m]")
            axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axs[2].scatter(np.array(tracks[str(i)])[:,0], np.array(tracks[str(i)])[:,3], s = size)
            axs[2].grid(True)
            axs[2].set_xlabel("Time [s]")
            axs[2].set_ylabel("$r_z$ [m]")
            axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            track_count += 1
            
plt.tight_layout()
axs[1].legend(bbox_to_anchor=(1.04,0.8), loc="upper left", borderaxespad=0,fontsize=14)  
#plt.savefig("test"+str(testnr)+"/mht_xyz_snr"+str(snr)+"_"+str(NFFT)[:2]+"k" + "wave_" + str(wavelet) + ".pdf")

plt.show()


'''
fig, axs = plt.subplots(3,1, sharex=True,sharey=False,figsize=(14,10))
fig.subplots_adjust(left=0.1, wspace=0.3)
fig.suptitle("Data, " + "SNR =" + str(snr) +", NFFT =" + str(NFFT)[:2]+"k",fontsize=29)   
axs[0].scatter(time_xyz[:,0], time_xyz[:,1], color="blue",alpha=0.7, s = size)
axs[0].grid(True)
axs[0].set_ylabel("$r_x$ [m]")

axs[1].scatter(time_xyz[:,0], time_xyz[:,2], color="blue",alpha=0.7, s = size)
axs[1].grid(True)
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axs[1].set_ylabel("$r_y$ [m]")

axs[2].scatter(time_xyz[:,0], time_xyz[:,3], color="blue",alpha=0.7, s = size)
axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axs[2].grid(True)
axs[2].set_xlabel("Time [s]")
axs[2].set_ylabel("$r_z$ [m]")

#plt.savefig("test"+str(testnr)+"/data"+str(testnr)+"_xyz_snr"+str(snr)+"_"+str(NFFT)[:2]+"k" + "wave_" + str(wavelet) + ".pdf")
plt.show()
'''

if testnr == 2:
    s_w = 0.001*np.eye(3) @ np.array([1]*3)
    m_pred = 2.01*np.eye(3)
    k_gain = m_pred @ np.linalg.inv(s_w + m_pred)
    corrected = dict()
    
    for key in all_predicts:
        # We cut off 3 indeces from track, cause weird stuff is hapenning
        # make sure it's the right points we cut off
    
        predict = np.array(all_predicts[key][:])
        if len(predict) >4:
            t = predict[:-1, 0]
            predict = predict[:-1, 1:]
            key = str(int(key))
            start = np.where(np.array(tracks[key])[:,0] == round(all_predicts[float(key)][0][0],1)-0.1)[0][0]
            end = np.where(np.array(tracks[key])[:,0] == round(all_predicts[float(key)][-2][0],1))[0][0]
            track = np.array(tracks[key])[start:end, 1:]
        
            # Transposing cuz dimensions are weird
            corrected[key] = np.column_stack((t, predict + (k_gain @ (track - predict).T).T))
        
    mse = {}
    dist_plot = {}
    txt_file = []
    for j,key in enumerate(corrected):
        correction_compare = True #if true then we compare corr and true orbit, otherwise we compare the data and true orb
        
        if correction_compare:
            # corrections vs true
            track_ = entire_orb[sat_nr-1]
            corrected_ = corrected[key]
        else:
            #data vs true
            corrected_ = entire_orb[sat_nr-1]
            track_ = time_xyz
        
        start = np.where(round(corrected[key][0][0],1) == track_)[0][0]
        
        #taking into account if corr or track has the highest time index
        if round(np.max(corrected_[:,0]),1) > round(np.max(track_[:,0]),1):
            end = np.where(corrected_ == round(track_[-1][0],1))[0][0]
            track_compare = track_[start:, :]
            corrected_ = corrected_[:end+1,:]
        else:
            end = np.where(round(corrected[key][-1][0],1) == track_)[0][0]
            track_compare = track_[start:end+1, :]
        
    
        mse[key] = 0
        dist_arr = np.zeros(len(track_compare))
        for i,(tra,corr) in enumerate(zip(track_compare,corrected_)):
            mse[key] += np.linalg.norm(tra[1:]-corr[1:])**2
            dist_arr[i] = np.linalg.norm(tra[1:]-corr[1:])
        
        mse[key] *= 1/len(track_compare)
        
        dist_plot[key] = dist_arr
        
        if correction_compare:
            #print(np.mean(dist_plot[key]))
            plt.plot(track_compare[:,0],dist_plot[key],color="b")
            plt.title("True Orbit vs Kalman (adj covariance + Wavelet)")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            #plt.savefig(path+test_number+f"_sat{sat_nr}_dist_snr{snr}_corr_adj-cov_wav.pdf")
            plt.show()
        else:
            #print(np.mean(dist_plot[key]))
            plt.plot(track_compare[:,0],dist_plot[key],color="b")
            plt.title("True Orbit vs Data")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            #plt.savefig(path+test_number+f"_sat{sat_nr}_dist_snr{snr}_data.pdf")
            plt.show()
        
        txt_file.append(f"Track number {sat_nr} MSE:\n{mse[key]}")

