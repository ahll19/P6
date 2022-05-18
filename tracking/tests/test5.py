import numpy as np
import matplotlib.pyplot as plt
import tracking as tr
from kalman_gating import KalmanGating as KG


def test(snr, nfft):
    snr_nfft_key = (snr, nfft)
    # Create the names to do the test with
    true_detections_names = dict()
    false_detections_names = dict()

    for _nfft in [15, 50]:
        for _snr in [10, 20, 50]:
            true_detections_names[(_snr, _nfft)] = [f"snr{snr}/truth{i}.txt" for i in range(1, 6)]
            false_detections_names[(_snr, _nfft)] = f"nfft_{nfft}k/false.txt"

    # import the data
    true_detections = []
    true_detections_times = []
    true_detections_minmax_times = []
    for track_num in true_detections_names[snr_nfft_key]:
        imported = np.array(tr.import_data(track_num)).T
        converted = tr.conversion(imported)

        t = imported[:, 0]
        true_detections.append(converted)
        true_detections_times.append(t)
        true_detections_minmax_times.append((np.min(t), np.max(t)))

    imported = np.array(tr.import_data(false_detections_names[snr_nfft_key])).T
    converted = tr.conversion(imported)
    false_detections = converted

    # combine and slice the data
    all_true_detections = np.concatenate(true_detections)
    all_detections = np.concatenate((all_true_detections, false_detections))
    all_detections_sliced = tr.time_slice(all_detections[all_detections[:, 0].argsort()])

    # initialize the kalman filters
    mults = [1, 1 / 50, 1]
    s_u, s_w, m_init = np.eye(6) * mults[0], np.eye(6) * mults[1], np.eye(6) * mults[2]
    filter_running = [False] * 5
    x_inits = [td[0, :].reshape(4, ) for td in true_detections]
    x_1s = [td[1, :].reshape(4, ) for td in true_detections]

    kalman_filters = [KG(s_u, s_w, x_init, m_init) for x_init in x_inits]
    for x_1, kf in zip(x_1s, kalman_filters):
        kf.init_gate(x_1)

    # Run through the Kalman filters
    for sweep in all_detections_sliced:
        sweep_time = sweep[0, 0]

        for i, kf in enumerate(kalman_filters):
            minmax_t = true_detections_minmax_times[i]
            if sweep_time == minmax_t[0]:
                filter_running[i] = True
            elif sweep_time == minmax_t[1]:
                filter_running[i] = False

            if filter_running[i]:
                kf.prediction(append_prediction=True)
                point = kf.gate(sweep)
                kf.observation(point)

    # Get the data from the filters
    filter_states = []
    for kf, t_lim in zip(kalman_filters, true_detections_minmax_times):
        state = np.array(kf.states_corr)[:, :3]
        t = np.linspace(t_lim[0], t_lim[1], len(state))
        txyz = np.column_stack((t, state))

        filter_states.append(txyz)

    results = {
        "filters": kalman_filters,
        "true_detections": true_detections,
        "false_detections": false_detections,
        "all_detections": all_detections,
        "tracks": filter_states
    }

    print(f"snr:{snr} - nfft{nfft}k - Done!")

    return results


# %%
entire_orbit_names = [f"snr50/entireOrbit{i}.txt" for i in range(1, 6)]
entire_orbit_dat = []
entire_orbit_t = []

for name in entire_orbit_names:
    dat, _, t = tr.velocity_algo(name)
    entire_orbit_dat.append(dat[:, :3])
    entire_orbit_t.append(t)
    print(f"Imported {name}")

# %%
snrs = [20]
nffts = [50]
results = []

_xyz = ["x", "y", "z"]
xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]

for snr in snrs:
    for nfft in nffts:
        res = test(snr, nfft)
        results.append((res, (snr, nfft)))


# %%
for result in results:
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728',
              u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
              u'#bcbd22', u'#17becf']
    ccolors = [u'#e0884b', u'#0080f1', u'#d35fd3', u'#29d8d7',
               u'#6b9842', u'#73a9b4', u'#1c883d', u'#808080',
               u'#4342dd', u'#e84130']
    tracks = result[0]["tracks"]
    detects = result[0]["true_detections"]
    all_detects = result[0]["all_detections"]
    false_detects = result[0]["false_detections"]
    snr, nfft = result[1]
    fig1, ax1 = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    for i in range(3):
        ax1[i].scatter(all_detects[:, 0], all_detects[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax1[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax1[i].set_ylabel(xyz[i])

    ax1[-1].set_xlabel("Time [s]")
    fig1.tight_layout()
    fig1.savefig(f"test5_figs/snr{result[1][0]}_nfft{result[1][1]}_{_xyz[i - 1]}_detects.pdf")

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    for i in range(1, 4):
        for j in range(5):
            ax[i - 1].scatter(false_detects[:, 0], false_detects[:, i], label="False detections", marker="x", s=1,
                              c='k', zorder=1, alpha=0.7)
            ax[i - 1].plot(tracks[j][:, 0], tracks[j][:, i], label=f"Track {j + 1}", c=ccolors[j], zorder=3, lw=0.8)
            ax[i - 1].scatter(detects[j][:, 0], detects[j][:, i], label=f"True detections {j + 1}", s=12, c=colors[j],
                              zorder=2, alpha=0.5)
            ax[i - 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax[i - 1].set_ylabel(xyz[i - 1])

    ax[-1].set_xlabel("Time [s]")
    handles, labels = ax[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 0.8), loc="upper left", borderaxespad=0,
               fontsize=14)
    fig.tight_layout()
    fig.savefig(f"test5_figs/snr{result[1][0]}_nfft{result[1][1]}.pdf", bbox_inches="tight")

    print("Saved plots!")

    distances = []
    for i, track in enumerate(tracks):
        track_dist = []
        for j, _t in enumerate(track[:, 0]):
            _t1, _t2 = np.round(_t, 1), np.round(entire_orbit_t[i], 1)
            where_res = np.where(_t1 == _t2)[0]
            if len(where_res) == 1:
                idx = int(where_res)

                dx = entire_orbit_dat[i][idx, 0] - track[j, 1]
                dy = entire_orbit_dat[i][idx, 1] - track[j, 2]
                dz = entire_orbit_dat[i][idx, 2] - track[j, 3]

                dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                track_dist.append(dist)

        distances.append(np.array(track_dist))

    for i, inf in enumerate(zip(tracks, distances)):
        track, dist = inf[0], inf[1]
        plt.plot(track[:, 0], dist)
        plt.xlabel(r"Time $[s]$")
        plt.ylabel(r"Distance $[m]$")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.savefig(f"test5_figs/snr{result[1][0]}_nfft{result[1][1]}_dist{i + 1}.pdf")
        plt.show()
    print("Saved distance plots")
    #
    # save_names = [f"test5_results/mse_snr{result[1][0]}_nfft{result[1][1]}_track{i}.txt" for i in range(1, 6)]
    # for i, dist in enumerate(distances):
    #     mse = np.sum(dist ** 2) / len(dist)
    #     with open(save_names[i], "w") as myfile:
    #         myfile.write(str(mse))
    #
    # print("Saved MSEs")
