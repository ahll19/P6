# Testen er udført med koden som den var ved commit:
#   1b63506736e85081a8b8d4e080ab7a6757cab0b0
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import importlib

sys.path.insert(1, os.getcwd())
import tracking as tr

importlib.reload(tr)

# %%
# specificer navne for test data
entire_names = ["snr10/entireOrbit" + str(i) + ".txt" for i in range(1, 6)]
detect_names_10 = ["snr10/truth" + str(i) + ".txt" for i in range(1, 6)]
detect_names_20 = ["snr20/truth" + str(i) + ".txt" for i in range(1, 6)]
detect_names_50 = ["snr50/truth" + str(i) + ".txt" for i in range(1, 6)]

# kør velocity algorithm på test dataen
entire_velocity = [tr.velocity_algo(name) for name in entire_names]
detect_velocity_10 = [tr.velocity_algo(name) for name in detect_names_10]
detect_velocity_20 = [tr.velocity_algo(name) for name in detect_names_20]
detect_velocity_50 = [tr.velocity_algo(name) for name in detect_names_50]

# %%
# sæt identitet til alle covariance gæt
cov_w, cov_u, M_initial = [np.eye(6)] * 3

# sæt initial guesses for x
x_init_10 = [res[0][0, :] for res in detect_velocity_10]
x_init_20 = [res[0][0, :] for res in detect_velocity_20]
x_init_50 = [res[0][0, :] for res in detect_velocity_50]

# sæt state, dt, og t for alle satelliter
state_10 = [res[0] for res in detect_velocity_10]
state_20 = [res[0] for res in detect_velocity_20]
state_50 = [res[0] for res in detect_velocity_50]
dt_10 = [res[1] for res in detect_velocity_10]
dt_20 = [res[1] for res in detect_velocity_20]
dt_50 = [res[1] for res in detect_velocity_50]
t_10 = [res[2] for res in detect_velocity_10]
t_20 = [res[2] for res in detect_velocity_20]
t_50 = [res[2] for res in detect_velocity_50]

# initialize kalman filtrerne
kalman_10 = []
kalman_20 = []
kalman_50 = []

for i in range(5):
    filter_10 = tr.Kalman(cov_u, cov_w, x_init_10[i], M_initial, dt_10[i])
    filter_20 = tr.Kalman(cov_u, cov_w, x_init_20[i], M_initial, dt_20[i])
    filter_50 = tr.Kalman(cov_u, cov_w, x_init_50[i], M_initial, dt_50[i])

    kalman_10.append(filter_10)
    kalman_20.append(filter_20)
    kalman_50.append(filter_50)

# %%
# kør simuleringen for alt dataen
for i in range(5):
    kalman_10[i].run_sim(state_10[i])
    kalman_20[i].run_sim(state_20[i])
    kalman_50[i].run_sim(state_50[i])

# %%
# plot dataen
for i in range(5):
    # hent data
    names_10, vals_10 = kalman_10[i].get_data()
    names_20, vals_20 = kalman_20[i].get_data()
    names_50, vals_50 = kalman_50[i].get_data()

    # lav og gem plots
    savestr_10 = "test2/track" + str(i + 1) + "_snr10"
    savestr_20 = "test2/track" + str(i + 1) + "_snr20"
    savestr_50 = "test2/track" + str(i + 1) + "_snr50"

    tr.plot_data(vals_10[0], t_10[i], "Kalman",
                 entire_velocity[i][0], entire_velocity[i][2], "True orbit",
                 1.5, savename=savestr_10)

    tr.plot_data(vals_20[0], t_20[i], "Kalman",
                 entire_velocity[i][0], entire_velocity[i][2], "True orbit",
                 1.5, savename=savestr_20)

    tr.plot_data(vals_50[0], t_50[i], "Kalman",
                 entire_velocity[i][0], entire_velocity[i][2], "True orbit",
                 1.5, savename=savestr_50)


# %%
# plot MSE og gem
mse_string_10 = ""
mse_string_20 = ""
mse_string_50 = ""

for i in range(5):
    # hent data
    names_10, vals_10 = kalman_10[i].get_data()
    names_20, vals_20 = kalman_20[i].get_data()
    names_50, vals_50 = kalman_50[i].get_data()

    # find MSE
    mse_10, dist_10 = tr.track_MSE(vals_10[1], entire_velocity[i][0], t_10[i], entire_velocity[i][2])
    mse_20, dist_20 = tr.track_MSE(vals_20[1], entire_velocity[i][0], t_20[i], entire_velocity[i][2])
    mse_50, dist_50 = tr.track_MSE(vals_50[1], entire_velocity[i][0], t_50[i], entire_velocity[i][2])

    # strings til at gemme plots og MSE data
    mse_string_10 += "SNR10 Track number " + str(i + 1) + " MSE: " + str(mse_10) + "\n"
    mse_string_20 += "SNR20 Track number " + str(i + 1) + " MSE: " + str(mse_20) + "\n"
    mse_string_50 += "SNR50 Track number " + str(i + 1) + " MSE: " + str(mse_50) + "\n"
    mse_save_string_10 = "test2/snr10_distanceplot_track" + str(i + 1) + ".pdf"
    mse_save_string_20 = "test2/snr20_distanceplot_track" + str(i + 1) + ".pdf"
    mse_save_string_50 = "test2/snr50_distanceplot_track" + str(i + 1) + ".pdf"

    # lav og gem plots
    plt.plot(dist_10)
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.savefig(mse_save_string_10)
    plt.show()

    plt.plot(dist_20)
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.savefig(mse_save_string_20)
    plt.show()

    plt.plot(dist_50)
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.savefig(mse_save_string_50)
    plt.show()

with open('test2/snr10_MSE.txt', "w") as myfile:
    myfile.write(mse_string_10)

with open('test2/snr20_MSE.txt', "w") as myfile:
    myfile.write(mse_string_20)

with open('test2/snr50_MSE.txt', "w") as myfile:
    myfile.write(mse_string_50)
