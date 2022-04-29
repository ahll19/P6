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
timesort_xyz = tr.time_slice(time_xyz)

# %% creating initial tracks
initial_track_keys = list(range(1, timesort_xyz[0].shape[1]+1))
tracks = {0: []}
for i in range(len(initial_track_keys)):
    _key = initial_track_keys[i]
    _point = timesort_xyz[0][0, i, 1:]
    tracks[_key] = _point

# %% Initial gating
hyp1_table = create_hyp_table(timesort_xyz[0][0, :, 1:], timesort_xyz[1][0, :, 1:], tracks, initial_hyp=True)

def N_pdf(mean,Sig_inv):
    Sig = np.linalg.inv(Sig_inv)
    n = len(Sig)
    return (np.exp(-0.5 * (mean.T@Sig_inv@mean))) / \
        (np.sqrt((2*np.pi)**n * np.linalg.norm(Sig)))

def beta_density(NFFT,n):
    n_FA = NFFT*np.exp(-10)
    beta_FT = n_FA/n
    beta_NT = (n-n_FA)/n
    return beta_FT, beta_NT

def Pik(H, c=1, P_g=0.2, P_D = 0.2, NFFT=15000, 
        y_t=None, y_t_hat=None, Sig_inv=None):
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
    
    beta_FT, beta_NT = beta_density(NFFT,len(H))
    N_TGT = np.max(H)-len(H) #Number of previously known targets
    
    prob = np.zeros(len(H[0]))
    for i,hyp in enumerate(H.T): 
        N_FT = np.count_nonzero(hyp==0) #Number of false 
        N_NT = np.count_nonzero(hyp>=N_TGT+1) #Number of known targets in hyp
        N_DT = len(hyp)-N_FT-N_NT #Number of prioer targets in given hyp

        #beta_FT = N_FT/(N_NT+N_DT+N_FT)
        #beta_NT = N_NT/(N_FT+N_DT+N_NT)

        prob[i] = (1/c) * (P_D**(N_DT) * (1-P_D)**(N_TGT-N_DT) * \
            beta_FT**(N_FT) * beta_NT**(N_NT))
        
        if N_DT >= 1:
            product = 1
            for j in range(N_DT): 
                product *= N_pdf(y_t[j]-y_t_hat[j],Sig_inv) # Må være den prediction der hører til givet punkt der menes
            
            prob[i] *= product*P_g
    
    prob_hyp = np.vstack((prob,H))
    
    prob_hyp = prob_hyp.T[prob_hyp.T[:,0].argsort()[::-1]].T
    
    return prob_hyp


def prune(prob_hyp,th=0.1,N_h=10000):
    cut_index = np.min(np.where(prob_hyp[0]<th))
    
    pruned_hyp = prob_hyp[:,:cut_index]
    
    if len(pruned_hyp[0]) >= N_h:
        pruned_hyp = pruned_hyp[:,:N_h]
    
    return pruned_hyp

P_D = 1 - np.exp(-10)

meh = Pik(hyp1_table, P_g = 0.2, P_D=P_D,c=0.00001)
print(meh)
print(prune(meh))


    

