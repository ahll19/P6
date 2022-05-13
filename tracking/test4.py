import numpy as np
import matplotlib.pyplot as plt
import tracking as tr
from kalman_gating import KalmanGating as KG

# %% Import the actual orbits
entire_orbits_name = ["snr50/entireOrbit" + str(i) + ".txt" for i in range(1, 6)]
_data = []
big_list = []
entire_data = []
entire_time = []

for i, file_ in enumerate(entire_orbits_name):
    _data.append(np.array(tr.import_data(file_)).T)
    _dat = np.array(tr.import_data(file_)).T
    _dat = tr.conversion(_dat)
    big_list.append(_dat)

    _dat = tr.velocity_algo(file_)
    t = _dat[2]
    r = _dat[0][:, :3]

    entire_data.append(r)
    entire_time.append(t)

data_ = np.concatenate(_data)
data_sliced_ = tr.time_slice(tr.conversion(data_[data_[:, 0].argsort()]))

# %% Slice the entire orbits, to the times we can see the satellites
track_time_names = [f"snr50/truth{i}.txt" for i in range(1, 6)]
track_times = [tr.import_data(name)[0] for name in track_time_names]
track_minmax_times = [(np.min(t), np.max(t)) for t in track_times]

data = []
times = []

for i, ent_time in enumerate(entire_time):
    minmax = track_minmax_times[i]
    idx_min = int(np.where(minmax[0] == ent_time)[0])
    idx_max = int(np.where(minmax[1] == ent_time)[0])

    data.append(entire_data[i][idx_min:idx_max, :])
    times.append(entire_time[i][idx_min:idx_max])

abs_time = (min([np.min(tt) for tt in track_times]), max([np.max(tt) for tt in track_times]))
max_iter, i = len(data_sliced_), 0
abs_min, abs_max = None, None

while i < max_iter and (abs_min is None or abs_max is None):
    _t = data_sliced_[i][0, 0]

    if _t == abs_time[0]:
        abs_min = i
    elif _t == abs_time[1]:
        abs_max = i

    i += 1

data_sliced = data_sliced_[abs_min:abs_max]

# %% Start 5 kalman filters apriori
mults = [1, 1 / 50, 1]
s_u, s_w, m_init = np.eye(6) * mults[0], np.eye(6) * mults[1], np.eye(6) * mults[2]
filter_running = [False] * 5

x_inits = [np.hstack((t[0], dat[0, :])) for t, dat in zip(times, data)]
x_1s = [np.hstack((t[1], dat[1, :])) for t, dat in zip(times, data)]

kalman_filters = [KG(s_u, s_w, x_init.reshape(4, ), m_init) for x_init in x_inits]
for kf, x_1 in zip(kalman_filters, x_1s):
    kf.init_gate(x_1)

for i, dat in enumerate(data_sliced):
    for j, kf in enumerate(kalman_filters):
        # Stop and start filters according to their track start and end times
        if track_minmax_times[j][0] == dat[0, 0]:
            filter_running[j] = True
        elif track_minmax_times[j][1] == dat[0, 0]:
            filter_running[j] = False

        if filter_running[j]:
            kf.prediction(append_prediction=True)
            point = kf.gate(dat)
            kf.observation(point)

# %% Plot the apriori data
track_color = ["r", "g", "b", "orange", "pink"]
track_coordinate = ["x", "y", "z"]
for i in range(3):
    for j, track in enumerate(data):
        state = np.array(kalman_filters[j].states_corr)

        plt.title(f"{track_coordinate[i]}")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.xlabel(r"$[s]$")
        plt.ylabel(r"$[m]$")
        plt.tight_layout()

        plt.plot(times[j], track[:, i], c=track_color[j], lw=10, alpha=0.2)
        plt.plot(times[j], state[1:, i], c=track_color[j], ls='-.')

    plt.show()
