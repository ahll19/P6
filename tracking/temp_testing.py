# Testen er udført med koden som den var ved commit:
#
import numpy as np
import os, sys
import matplotlib.pyplot as plt
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

test1_states, test1_dt, _ = [], [], 0
for i in range(len(test1_names)):
    _state, _dt, _ = test1_results[i]
    test1_states.append(_state)
    test1_dt.append(_dt)

test2_states, test2_dt, test2_names_flattenned, _ = [], [], [], 0
for i in range(len(test2_names)):
    for j in range(len(test2_names[0])):
        _state, _dt, _ = test2_results[i][j]
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
# Plot dataen
fig_save_names = ["snr10", "snr20", "snr50"]

# plot track 1
for i in range(3):
    names1, vals1 = kalman_filters_1[0].get_data()
    names2, vals2 = kalman_filters_2[2*i].get_data()

    time1 = tr.import_data(test1_names[0])[0]
    time2 = tr.import_data(test2_names[i][0])[0]

    tr.plot_data(vals2[0], time2, "Kalman",  vals1[0], time1,
                 "True orbit", 1.5, savename="test1/track1"+fig_save_names[i])

# plot track 2
for i in range(3):
    names1, vals1 = kalman_filters_1[1].get_data()
    names2, vals2 = kalman_filters_2[2*i+1].get_data()

    time1 = tr.import_data(test1_names[1])[0]
    time2 = tr.import_data(test2_names[i][1])[0]

    tr.plot_data(vals2[0], time2, "Kalman",  vals1[0], time1,
                 "True orbit", 1.5, savename="test1/track2"+fig_save_names[i])

#%%
# kig på MSE for vores tracks
for i in range(3):
    names1, vals1 = kalman_filters_1[0].get_data()
    names2, vals2 = kalman_filters_2[2*i].get_data()

    time1 = tr.import_data(test1_names[0])[0]
    time2 = tr.import_data(test2_names[i][0])[0]
    time2 = time2[1:]

    mse, square_diff = tr.track_MSE(vals2[1], vals1[1], time2, time1)
    plt.plot(square_diff[10:])
    plt.title("Satellite 1: " + fig_save_names[i])
    plt.show()
    print(mse)


for i in range(3):
    names1, vals1 = kalman_filters_1[1].get_data()
    names2, vals2 = kalman_filters_2[2*i+1].get_data()

    time1 = tr.import_data(test1_names[1])[0]
    time2 = tr.import_data(test2_names[i][1])[0]
    time2 = time2[1:]

    mse, square_diff = tr.track_MSE(vals2[1], vals1[1], time2, time1)
    plt.plot(square_diff[10:])
    plt.title("Satellite 2: " + fig_save_names[i])
    plt.show()
    print(mse)

