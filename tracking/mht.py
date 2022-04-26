import numpy as np
import importlib
import tracking as tr
import sys, os
sys.path.insert(1, os.getcwd())

importlib.reload(tr)

# %% inputs
vmax = 12000

# %% data import and sort

imports = ["snr50/truth1.txt","snr50/truth2.txt","nfft_15k/false.txt"]

_data = []
for i, file_ in enumerate(imports):
    _data.append(np.array(tr.import_data(file_)).T)

    if i == 0:
        _data[0][:,0] = np.array(_data[0][:,0])-5
    if i == 2:
        _data[2][:,0] = np.array(_data[2][:,0])+5

data_ = np.concatenate((_data[0],_data[1]))
data_ = np.concatenate((data_,_data[2]))
data = data_[data_[:,0].argsort()]

data = data[:12]
#Convert to cartesian coordinates
time_xyz = tr.conversion(data)
timesort_xyz = tr.time_slice(time_xyz)


# %% functions
def init_gate(q1, q2, dt, vmax=12000):
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


# %% Initial gating

d_hyp = {}
hyp_individual = []
for i in range(len(timesort_xyz[0][0])):
    init_hyp = [0, i + 1]

    for j in range(len(timesort_xyz[1][0])):
        if init_gate(timesort_xyz[0][0][i][1:], timesort_xyz[1][0][j][1:],
                     timesort_xyz[1][0][0][0] - timesort_xyz[0][0][0][0]):
            init_hyp.append(-np.inf)

    hyp_individual.append(init_hyp)

d_hyp.update({1:hyp_individual})
#hyp = [init_hyp]
for sat in timesort_xyz[2:]:
    number_obs = len(sat[0])
    for sat_i in sat:
        a = 1
        # print(number_obs, sat_i)

