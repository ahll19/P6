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
    for track_idx, track in enumerate(tracks):
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
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
        ax[-1].set_xlabel(r"Time $[s]$")
        colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
        c_colors = [u'#e0884b', u'#0080f1', u'#d35fd3', u'#29d8d7', u'#6b9842', u'#73a9b4', u'#1c883d', u'#808080', u'#4342dd', u'#e84130']
        for i in range(3):
            ax[i].scatter(data_converted[:, 0], data_converted[:, i+1], s=12, zorder=1, label="Detections", c=colors[track_idx], alpha=0.5)
            ax[i].plot(data_converted[:-1, 0], state[:, i], lw=0.8, zorder=2, label=f"Track {track_idx+1}", c=c_colors[track_idx])
            ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax[i].set_ylabel(xyz[i])
        handles, labels = ax[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 0.8), loc="upper left", borderaxespad=0, fontsize=14)
        fig.tight_layout()
        fig.savefig(f"test2_figs/track{track}_snr{snr}.pdf", bbox_inches="tight")
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
