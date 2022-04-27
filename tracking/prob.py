import numpy as np
from itertools import product


def init_gate(q1, q2, dt, vmax=0.5):
    if np.linalg.norm(q1 - q2) <= vmax * dt:
        return True
    else:
        return False
    
# simulate data
m01 = np.random.rand(3)
m02 = np.random.rand(3)
m03 = np.random.rand(3)
S0 = [m01, m02, m03]
keylist = list(range(1, len(S0)+1))

tracks = {0: []}
for i in range(len(keylist)):
    tracks[keylist[i]] = [S0[i]]


m11 = np.random.rand(3)
m12 = np.random.rand(3)
m13 = np.random.rand(3)
S1 = [m11, m12, m13]

# create initial hypothesis table
hyp_table = []
for i in range(len(S1)):
    mn_hyp = [0]
    for j in range(len(S0)):
        if init_gate(S0[i], S1[j], 1, 0.75):
            mn_hyp.append(j+1)
    
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

def N_pdf(mean,Sig_inv):
    Sig = np.linalg.inv(Sig_inv)
    n = len(Sig)
    return (-0.5 * (mean.T@Sig_inv@mean)) / \
        (np.sqrt((2*np.pi)**n * np.linalg.norm(Sig)))

def Pik(H, c=1, P_g=0.2, P_D = 0.2, prior_info=False, 
        y_t=[], y_t_hat=[], Sig_inv=[]):
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
    
    N_TGT = np.max(H)-len(H) #Number of previously known targets
    
    prob = np.zeros(H.T[0])
    for i,hyp in enumerate(H.T): 
        N_FT = np.count_nonzero(hyp==0) #Number of false 
        N_NT = np.count_nonzero(hyp>=N_TGT+1) #Number of known targets in hyp
        N_DT =  len(hyp)-N_FT-N_NT #Number of prioer targets in given hyp
        
        if N_NT+N_DT == 0:
            beta_FT = 1
        else:
            beta_FT = N_FT/(N_NT+N_DT)
        
        if N_FT+N_DT==0:
            beta_NT = 1
        else:
            beta_NT = N_NT/(N_FT+N_DT)
        
        prob[i] = (1/c) * (P_D**(N_DT) * (1-P_D)**(N_TGT-N_DT) * \
            beta_FT**(N_FT) * beta_NT**(N_NT))
        
        if prior_info == True:
            product = 1
            for i in range(N_DT): 
                product *= N_pdf(y_t-y_t_hat,Sig_inv) # ved ikke om det skal være dobbelt for loop ift alle punkter y_t til tiden t
            
            prob[i] *= product*P_g
    
    return prob



