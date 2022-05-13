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
    mults = [1, 1/50, 1]
    s_u, s_w, m_init = np.eye(6) * mults[0], np.eye(6) * mults[1], np.eye(6) * mults[2]
    filter_running = [False]*5
    x_inits = [td[0, :].reshape(4,) for td in true_detections]
    x_1s = [td[1, :].reshape(4,) for td in true_detections]

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


if __name__ == "__main__":
    snrs = [10, 20, 50]
    nffts = [15, 50]
    results = []
    xyz = [r"$r_x$", r"$r_y$", r"$r_z$"]

    for snr in snrs:
        for nfft in nffts:
            res = test(snr, nfft)
            results.append((res, (snr, nfft)))

    for result in results:
        tracks = result[0]["tracks"]
        detects = result[0]["true_detections"]
        all_detects = result[0]["all_detections"]
        snr, nfft = result[1]
        for i in range(3):
            plt.scatter(all_detects[:, 0], all_detects[:, i+1], marker="x", s=0.2, alpha=0.4, c='k')
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.xlabel("Time [s]")
            plt.ylabel(xyz[i])
            plt.title(f"SNR: {snr} - NFFT: {nfft} - All detections")
            plt.show()

        for i in range(1, 4):
            for j in range(5):
                plt.plot(tracks[j][:, 0], tracks[j][:, i], label="Tracks", c='r', zorder=3)
                plt.scatter(detects[j][:, 0], detects[j][:, i], label="TD", s=2, c='b', zorder=2)
                plt.scatter(all_detects[:, 0], all_detects[:, i], label="FD", marker="x", s=0.2,
                            alpha=0.4, c='k', zorder=1)
                plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                plt.xlabel("Time [s]")
                plt.ylabel(xyz[i-1])
                plt.title(f"SNR: {snr} - NFFT: {nfft}")

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.show()

