# Testen er udført med koden som den var ved commit:
#
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import importlib

sys.path.insert(1, os.getcwd())
import tracking as tr

importlib.reload(tr)

# %%
# specificer navne for test data
entire_orbit_names = ["snr10/entireOrbit" + str(i) + ".txt" for i in range(1, 6)]
detection_orbit_names = ["snr10/truth" + str(i) + ".txt" for i in range(1, 6)]

# kør velocity algorithm på test dataen
entire_orbit_velocity = [tr.velocity_algo(name) for name in entire_orbit_names]
detection_orbit_velocity = [tr.velocity_algo(name) for name in detection_orbit_names]

# %%
# skær dataen til de tider vi skal bruge
times = []
entire_state = []
entire_dt = []
entire_t = []
for i in range(5):
    # hiv tids-vektor ud
    t_detect = detection_orbit_velocity[i][2]
    t_entire = entire_orbit_velocity[i][2]

    # find index for de tider vi skal bruge
    t_min, t_max = np.min(t_detect), np.max(t_detect)
    n1 = np.where(t_entire == t_min)[0][0]
    n2 = np.where(t_entire == t_max)[0][0]

    # skær dataen til givne indexer
    entire_state.append(entire_orbit_velocity[i][0][n1:n2])
    entire_dt.append(entire_orbit_velocity[i][1][n1:n2])
    entire_t.append(entire_orbit_velocity[i][2][n1:n2])

# %%
# sæt identitet til alle covariance gæt
cov_w, cov_u, M_initial = [np.eye(6)] * 3

kalman_filers = []
for i in range(5):
    # sæt initial gæt til første datapunkt, og bestem dt
    x = entire_state[i][0, :]
    dt = entire_dt[i]

    _kal_temp = tr.Kalman(cov_u, cov_w, x, M_initial, dt)
    kalman_filers.append(_kal_temp)

# %%
# kør simuleringen for alt dataen
for i in range(5):
    kalman_filers[i].run_sim(entire_state[i])

# %%
# plot dataen
for i in range(5):
    names, vals = kalman_filers[i].get_data()

    savestr = "test1/track" + str(i + 1) + ".pdf"
    tr.plot_data(vals[0], entire_t[i], "Kalman", entire_state[i], entire_t[i], "True orbit", 1.5, savename=savestr)

# %%
# plot MSE og gem
mse_string = ""
for i in range(5):
    names, vals = kalman_filers[i].get_data()

    mse, dist = tr.track_MSE(vals[1], entire_state[i], entire_t[i], entire_t[i])

    mse_string += "Track number " + str(i + 1) + " MSE: " + str(mse) + "\n"
    mse_save_string = "test1/distanceplot_track" + str(i + 1) + ".pdf"

    plt.plot(dist)
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.savefig(mse_save_string)
    plt.show()

with open('test1/MSE.txt', "w") as myfile:
    myfile.write(mse_string)
