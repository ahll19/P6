import numpy as np
import matplotlib.pyplot as plt
import tracking as tr

# %% Impot the data which is needed for the plots pt. 1
entire_orbits_name = ["snr50/entireOrbit" + str(i) + ".txt" for i in range(1, 6)]
_data = []
entire_data = []
entire_time = []

for i, file_ in enumerate(entire_orbits_name):
    _data.append(np.array(tr.import_data(file_)).T)
    _dat = np.array(tr.import_data(file_)).T
    _dat = tr.conversion(_dat)

    _dat = tr.velocity_algo(file_)
    t = _dat[2]
    r = _dat[0][:, :3]

    entire_data.append(r)
    entire_time.append(t)

data_ = np.concatenate(_data)
data_sliced_ = tr.time_slice(tr.conversion(data_[data_[:, 0].argsort()]))
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

entire_orbit = data_sliced_[abs_min:abs_max]

# %% Impot the data which is needed for the plots pt. 2
true_detection_10_names = [f"snr10/truth{i}.txt" for i in range(1, 6)]
true_detection_20_names = [f"snr20/truth{i}.txt" for i in range(1, 6)]
true_detection_50_names = [f"snr50/truth{i}.txt" for i in range(1, 6)]
false_detections_names = [f"nfft_{k}k/false.txt" for k in [15, 50]]

true_detections_10 = []
true_detections_20 = []
true_detections_50 = []

for i, name in enumerate(true_detection_10_names):
    imported = np.copy(np.array(tr.import_data(name)).T)
    converted = tr.conversion(np.copy(imported))
    true_detections_10.append(converted)

for i, name in enumerate(true_detection_20_names):
    imported = np.copy(np.array(tr.import_data(name)).T)
    converted = tr.conversion(np.copy(imported))
    true_detections_20.append(converted)

for i, name in enumerate(true_detection_50_names):
    imported = np.copy(np.array(tr.import_data(name)).T)
    converted = tr.conversion(np.copy(imported))
    true_detections_50.append(converted)

imported = np.copy(np.array(tr.import_data(false_detections_names[0])).T)
converted = tr.conversion(np.copy(imported))
false_detections_15 = converted

imported = np.copy(np.array(tr.import_data(false_detections_names[1])).T)
converted = tr.conversion(np.copy(imported))
false_detections_50 = converted

"""
The data is saved in the variables:
    data (list of clean tracks)
        times (times corresponding to data)
    true_detections_#
    false_detections_#
"""

# %% Plot test 1
for j in range(5):
    xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    for i in range(3):
        ax[i].scatter(times[j], data[j][:, i], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])

    ax[-1].set_xlabel(r"Time $[s]$")
    fig.tight_layout()
    fig.savefig(f"test_data_plots/test1_track{j + 1}.pdf", bbox_inches="tight")

# %% Plot test 2
for j in range(5):
    xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    for i in range(3):
        ax[i].scatter(true_detections_10[j][:, 0], true_detections_10[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])

    ax[-1].set_xlabel(r"Time $[s]$")
    fig.tight_layout()
    fig.savefig(f"test_data_plots/test2_track{j + 1}_snr10.pdf", bbox_inches="tight")

for j in range(5):
    xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    for i in range(3):
        ax[i].scatter(true_detections_20[j][:, 0], true_detections_20[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])

    ax[-1].set_xlabel(r"Time $[s]$")
    fig.tight_layout()
    fig.savefig(f"test_data_plots/test2_track{j + 1}_snr20.pdf", bbox_inches="tight")

for j in range(5):
    xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    for i in range(3):
        ax[i].scatter(true_detections_50[j][:, 0], true_detections_50[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])

    ax[-1].set_xlabel(r"Time $[s]$")
    fig.tight_layout()
    fig.savefig(f"test_data_plots/test2_track{j + 1}_snr50.pdf", bbox_inches="tight")

# %% Plot test 3
for j in range(5):
    xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    for i in range(3):
        ax[i].scatter(true_detections_10[j][:, 0], true_detections_10[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].scatter(false_detections_15[:, 0], false_detections_15[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
    ax[-1].set_xlabel(r"Time $[s]$")
    fig.tight_layout()
    fig.savefig(f"test_data_plots/test3_track{j + 1}_snr10_nfft15.pdf", bbox_inches="tight")

for j in range(5):
    xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    for i in range(3):
        ax[i].scatter(true_detections_10[j][:, 0], true_detections_10[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].scatter(false_detections_50[:, 0], false_detections_50[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
    ax[-1].set_xlabel(r"Time $[s]$")
    fig.tight_layout()
    fig.savefig(f"test_data_plots/test3_track{j + 1}_snr10_nfft50.pdf", bbox_inches="tight")

for j in range(5):
    xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    for i in range(3):
        ax[i].scatter(true_detections_20[j][:, 0], true_detections_20[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].scatter(false_detections_15[:, 0], false_detections_15[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
    ax[-1].set_xlabel(r"Time $[s]$")
    fig.tight_layout()
    fig.savefig(f"test_data_plots/test3_track{j + 1}_snr20_nfft15.pdf", bbox_inches="tight")

for j in range(5):
    xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    for i in range(3):
        ax[i].scatter(true_detections_20[j][:, 0], true_detections_20[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].scatter(false_detections_50[:, 0], false_detections_50[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
    ax[-1].set_xlabel(r"Time $[s]$")
    fig.tight_layout()
    fig.savefig(f"test_data_plots/test3_track{j + 1}_snr20_nfft50.pdf", bbox_inches="tight")

for j in range(5):
    xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    for i in range(3):
        ax[i].scatter(true_detections_50[j][:, 0], true_detections_50[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].scatter(false_detections_15[:, 0], false_detections_15[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
    ax[-1].set_xlabel(r"Time $[s]$")
    fig.tight_layout()
    fig.savefig(f"test_data_plots/test3_track{j + 1}_snr50_nfft15.pdf", bbox_inches="tight")

for j in range(5):
    xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    for i in range(3):
        ax[i].scatter(true_detections_50[j][:, 0], true_detections_50[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].scatter(false_detections_50[:, 0], false_detections_50[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
    ax[-1].set_xlabel(r"Time $[s]$")
    fig.tight_layout()
    fig.savefig(f"test_data_plots/test3_track{j + 1}_snr50_nfft50.pdf", bbox_inches="tight")

# %% Plot test 4
xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
for i in range(3):
    for j in range(5):
        ax[i].scatter(times[j], data[j][:, i], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
ax[-1].set_xlabel(r"Time $[s]$")
fig.tight_layout()
fig.savefig(f"test_data_plots/test4.pdf", bbox_inches="tight")

# %% Plot test 5
xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
for j in range(5):
    for i in range(3):
        ax[i].scatter(true_detections_10[j][:, 0], true_detections_10[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].scatter(false_detections_15[:, 0], false_detections_15[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
ax[-1].set_xlabel(r"Time $[s]$")
fig.tight_layout()
fig.savefig(f"test_data_plots/test5_snr10_nfft15.pdf", bbox_inches="tight")

xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
for j in range(5):
    for i in range(3):
        ax[i].scatter(true_detections_10[j][:, 0], true_detections_10[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].scatter(false_detections_50[:, 0], false_detections_50[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
ax[-1].set_xlabel(r"Time $[s]$")
fig.tight_layout()
fig.savefig(f"test_data_plots/test5_snr10_nfft50.pdf", bbox_inches="tight")

xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
for j in range(5):
    for i in range(3):
        ax[i].scatter(true_detections_20[j][:, 0], true_detections_20[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].scatter(false_detections_15[:, 0], false_detections_15[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
ax[-1].set_xlabel(r"Time $[s]$")
fig.tight_layout()
fig.savefig(f"test_data_plots/test5_snr20_nfft15.pdf", bbox_inches="tight")

xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
for j in range(5):
    for i in range(3):
        ax[i].scatter(true_detections_20[j][:, 0], true_detections_20[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].scatter(false_detections_50[:, 0], false_detections_50[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
ax[-1].set_xlabel(r"Time $[s]$")
fig.tight_layout()
fig.savefig(f"test_data_plots/test5_snr20_nfft50.pdf", bbox_inches="tight")

xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
for j in range(5):
    for i in range(3):
        ax[i].scatter(true_detections_50[j][:, 0], true_detections_50[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].scatter(false_detections_15[:, 0], false_detections_15[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
ax[-1].set_xlabel(r"Time $[s]$")
fig.tight_layout()
fig.savefig(f"test_data_plots/test5_snr50_nfft15.pdf", bbox_inches="tight")

xyz = [r"$r_x\ [m]$", r"$r_y\ [m]$", r"$r_z\ [m]$"]
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
for j in range(5):
    for i in range(3):
        ax[i].scatter(true_detections_50[j][:, 0], true_detections_50[j][:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].scatter(false_detections_50[:, 0], false_detections_50[:, i + 1], marker="x", s=1, c='k', alpha=0.7)
        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_ylabel(xyz[i])
ax[-1].set_xlabel(r"Time $[s]$")
fig.tight_layout()
fig.savefig(f"test_data_plots/test5_snr50_nfft50.pdf", bbox_inches="tight")
