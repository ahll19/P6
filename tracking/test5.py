import numpy as np
import matplotlib.pyplot as plt
import tracking as tr
from kalman_gating import KalmanGating as KG


# %% Import the data
snr_values = ["10", "20", "50"]
nfft_values = [15, 50]
true_detection_names = [[f"snr{val}/truth{i}.txt" for i in range(1, 6)] for val in snr_values]
false_detection_names = [f"nfft_{k}k/false.txt" for k in nfft_values]

track_times = [[tr.import_data(name)[0] for name in track] for track in true_detection_names]
track_minmax_times = [[(np.min(t), np.max(t)) for t in snr_time] for snr_time in track_times]

true_detections = []
false_detections = []

for i, snr_list in enumerate(true_detection_names):
    snr_detections = []
    for j, file_name in enumerate(snr_list):
        _imported = np.array(tr.import_data(file_name)).T
        _t = _imported[:, 0]
        _converted = tr.conversion(_imported)
        snr_detections.append(_converted)

    true_detections.append(snr_detections)

for i, file_name in enumerate(false_detection_names):
    _imported = np.array(tr.import_data(file_name)).T
    _t = _imported[:, 0]
    _converted = tr.conversion(_imported)
    false_detections.append(_converted)

# %% Combine the data into time-sliced sets
test_data_names = []
test_data_placeholder = []
test_data = []

for i in range(3):
    for j in range(2):
        _td = np.concatenate(true_detections[i])
        _detects = np.concatenate((_td, false_detections[j]))
        test_data_placeholder.append(_detects)

        test_data_names.append(f"snr{snr_values[i]}_nfft{nfft_values[j]}")

for tdp in test_data_placeholder:
    _tdp = tdp[tdp[:, 0].argsort()]
    test_data.append(tr.time_slice(_tdp))

# %% Initialize the Kalman filters
mults = [1, 1/50, 1]
s_u, s_w, m_init = np.eye(6) * mults[0], np.eye(6) * mults[1], np.eye(6) * mults[2]
filter_running = [[False]*5]*3

x_inits = [[track[0, :].reshape(4,) for track in snr_detect] for snr_detect in true_detections]
x_1s = [[track[1, :].reshape(4,) for track in snr_detect] for snr_detect in true_detections]

kalman_filters = [[KG(s_u, s_w, x_init, m_init) for x_init in snr_inits] for snr_inits in x_inits]
for i, snr_kf in enumerate(kalman_filters):
    for j, kf in enumerate(snr_kf):
        kf.init_gate(x_1s[i][j].reshape(4,))

# %% Run the kalman filters
num_runs = 113220
counter = 0
for dat in test_data:
    for meas in dat:
        for i, snr_kf in enumerate(kalman_filters):
            for j, kf in enumerate(snr_kf):
                if track_minmax_times[i][j][0] == meas[0, 0]:
                    filter_running[i][j] = True
                elif track_minmax_times[i][j][1] == meas[0, 0]:
                    filter_running[i][j] = False
                if filter_running[i][j]:
                    kf.prediction(append_prediction=True)
                    point = kf.gate(dat)
                    kf.observation(point)

    print(f"{counter*100/num_runs:.2f}%")
