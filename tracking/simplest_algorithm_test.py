# Testen er udført med koden som den var ved commit:
#   724ebfaf0f039a6a0e08aac2c3fca2683ba50662
import numpy as np
import os, sys
import importlib
sys.path.insert(1, os.getcwd())
import tracking as tr
importlib.reload(tr)


#%%
# hent dataen ned og kør velocity algorithm på dem
test1_names = [
    "snr10/entireOrbit1.txt",
    "snr10/entireOrbit2.txt"
]
test2_names = [
    ["snr10/truth1.txt", "snr10/truth2.txt"],
    ["snr20/truth1.txt", "snr20/truth2.txt"],
    ["snr50/truth1.txt", "snr50/truth2.txt"]
]

test1_results = [tr.velocity_algo(name) for name in test1_names]
test2_results = [[tr.velocity_algo(name) for name in snr] for snr in test2_names]

#%%
# Initialize Kalman like a dumbass

test1_states, test1_dt = [], []
for i in range(len(test1_names)):
    _state, _dt = test1_results[i]
    test1_states.append(_state)
    test1_dt.append(_dt)

test2_states, test2_dt, test2_names_flattenned = [], [], []
for i in range(len(test2_names)):
    for j in range(len(test2_names[0])):
        _state, _dt = test2_results[i][j]
        test2_states.append(_state)
        test2_dt.append(_dt)
        test2_names_flattenned.append(test2_names[i][j])

cov_w, cov_u = [np.eye(6)]*2
x_initial_guess, M_initial_guess = np.ones(6), np.eye(6)

kalman_filters_1 = [tr.Kalman(cov_u, cov_w, x_initial_guess, M_initial_guess, dt) for dt in test1_dt]
kalman_filters_2 = [tr.Kalman(cov_u, cov_w, x_initial_guess, M_initial_guess, dt) for dt in test2_dt]

#%%
# Kør kalman koden for alle filter
for i in range(len(kalman_filters_1)):
    kalman_filters_1[i].run_sim(test1_states[i])

for i in range(len(kalman_filters_2)):
    kalman_filters_2[i].run_sim(test2_states[i])

#%%
# Plot dataen, og gem den i en mappe så vi ikke skal køre filtret hver gang
test1_save_names = [
    "snr10_result/entireOrbit1.txt",
    "snr10_result/entireOrbit2.txt"
]
test2_save_names = [
    ["snr10_result/truth1.txt", "snr10/truth2.txt"],
    ["snr20_result/truth1.txt", "snr20/truth2.txt"],
    ["snr50_result/truth1.txt", "snr50/truth2.txt"]
]

# plot og gem track 1
for i in range(3):
    names1, vals1 = kalman_filters_1[0].get_data()
    names2, vals2 = kalman_filters_2[2*i].get_data()

    time1 = np.round(np.cumsum(test1_dt[0]), 1)
    time2 = np.round(np.cumsum(test2_dt[i*2]), 1)

    tr.plot_data(vals2[0][10:, :], time2, "Kalman", vals1[0][50:, :], time1, "True orbit", 1.5)

# plot og gem track 2
for i in range(3):
    names1, vals1 = kalman_filters_1[1].get_data()
    names2, vals2 = kalman_filters_2[2*i+1].get_data()

    time1 = np.round(np.cumsum(test1_dt[1]), 1)
    time2 = np.round(np.cumsum(test2_dt[i*2+1]), 1)

    tr.plot_data(vals2[0][10:, :], time2, "Kalman", vals1[0][50:, :], time1, "True orbit", 1.5)
