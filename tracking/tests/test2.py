import numpy as np
import matplotlib.pyplot as plt
import kalman_gating as kg
import tracking as tr


snrs = [10, 20, 50]
tracks = list(range(1, 6))
entire_orbit_converted_dict = dict()

for track in tracks:
    entire_obit_name = f"snr50/entireOrbit{track}.txt"
    entire_orbit_import = np.array(tr.import_data(entire_obit_name)).T
    entire_orbit_converted_dict[track] = tr.conversion(entire_orbit_import)

print("Imported data")

for snr in snrs:
    for track in tracks:
        entire_orbit_converted = entire_orbit_converted_dict[track]
        data_name = f"snr{snr}/truth{track}.txt"
        data_import = np.array(tr.import_data(data_name)).T
        data_converted = tr.conversion(data_import)
        mults = [1, 1/25, 1]
        su, sw, m_init = [np.eye(6) * m for m in mults]
        x_init, x_1 = data_converted[0], data_converted[1]
        kf = kg.KalmanGating(su, sw, x_init, m_init)
        kf.init_gate(x_1)
        for j in range(2, data_converted.shape[0]):
            kf.prediction(append_prediction=True)
            point = kf.gate([data_converted[j]])
            kf.observation(point)

        state = np.array(kf.states_corr)[:, :3]
        _xyz = ["x", "y", "z"]
        xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
        for i in range(3):
            plt.scatter(data_converted[:, 0], data_converted[:, i+1], s=4, zorder=1, label="Detections", c='b')
            plt.plot(data_converted[:-1, 0], state[:, i], lw=0.6, zorder=2, label="Track", c='r')
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.xlabel(r"Time $[s]$")
            plt.ylabel(xyz[i])
            plt.savefig(f"test2_figs/track{track}_snr{snr}_{_xyz[i]}.pdf")
            plt.show()

        equal_times = []
        equal_idxs = []
        for t in data_converted[:, 0]:
            _idx = np.where(t == entire_orbit_converted[:, 0])[0]
            if len(_idx) == 1:
                idx = int(_idx)
                equal_idxs.append(idx)
                equal_times.append(t)
        _ = equal_idxs.pop(0)
        equal_idxs = np.array(equal_idxs)

        diff = state - entire_orbit_converted[equal_idxs, 1:]
        dist = np.sqrt(np.sum(diff**2, axis=1))
        mse = np.sum(dist**2)/len(dist)

        plt.plot(equal_times[1:], dist)
        plt.xlabel(r"Time $[s]$")
        plt.ylabel(r"Distance $[m]$")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.savefig(f"test2_figs/track{track}_snr{snr}_mse.pdf")
        plt.show()

        with open(f"test2_results/snr{snr}_track{track}_mse.txt", "w") as my_file:
            my_file.write(str(mse))

        print(f"Did snr {snr} for track {track}")
