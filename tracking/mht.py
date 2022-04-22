import numpy as np
import importlib
import tracking as tr
importlib.reload(tr)

#%% inputs
vmax = 12000

#%% data import and sort

imports = ["snr50/truth1.txt","snr50/truth2.txt"]

_data = []
for i,file_ in enumerate(imports):
    _data.append(np.array(tr.import_data(file_)).T)

data_ = np.concatenate((_data[0],_data[1]))
data = data_[data_[:,0].argsort()]

#Convert to cartesian coordinates
time_xyz = tr.conversion(data[:,2], data[:,3], data[:,1])

#%% Initial gate
def init_gate(q1,q2,dt,vmax=12000):
    if np.linalg.norm(q1-q2)<= vmax*dt:
        return True
    else:
        return False
    



'''
test_names = ["snr50/truth1.txt"]

test_results = tr.velocity_algo(test_names[0])

#%%
cov_w, cov_u = [np.eye(6)]*2
x_initial_guess, M_initial_guess = np.ones(6), np.eye(6)

state = test_results[0]
dt = test_results[1]
t = test_results[2]

kalman = tr.Kalman(cov_u, cov_w, x_initial_guess, M_initial_guess, dt)
rvec_time = np.hstack((state[:,:3],np.array([t[:-1]]).T))

init_gate_th = vmax*round(np.diff(rvec_time[:,3][:2])[0],2)
for i in range(10):
    
    kalman.run_sim_mht(state[i])
    
'''   