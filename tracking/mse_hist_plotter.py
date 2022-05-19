import tracking as tr
import numpy as np
import matplotlib.pyplot as plt

# %%
test2_10_names = [f"tests/test2_results/snr10_track{i}_mse.txt" for i in range(1, 6)]
test2_20_names = [f"tests/test2_results/snr20_track{i}_mse.txt" for i in range(1, 6)]
test2_50_names = [f"tests/test2_results/snr50_track{i}_mse.txt" for i in range(1, 6)]

test3_10_15k_names = [f"tests/test3_results/MSE_sat{i}_snr10_nfft15k.txt" for i in range(1, 6)]
test3_10_50k_names = [f"tests/test3_results/MSE_sat{i}_snr10_nfft50k.txt" for i in range(1, 6)]
test3_20_15k_names = [f"tests/test3_results/MSE_sat{i}_snr20_nfft15k.txt" for i in range(1, 6)]
test3_20_50k_names = [f"tests/test3_results/MSE_sat{i}_snr20_nfft50k.txt" for i in range(1, 6)]
test3_50_15k_names = [f"tests/test3_results/MSE_sat{i}_snr50_nfft15k.txt" for i in range(1, 6)]
test3_50_50k_names = [f"tests/test3_results/MSE_sat{i}_snr50_nfft50k.txt" for i in range(1, 6)]

test4_names = [f"tests/test4_results/track{i}_mse.txt" for i in range(1, 6)]

test5_10_15k_names = [f"tests/test5_results/mse_snr10_nfft15_track{i}.txt" for i in range(1, 6)]
test5_10_50k_names = [f"tests/test5_results/mse_snr10_nfft50_track{i}.txt" for i in range(1, 6)]
test5_20_15k_names = [f"tests/test5_results/mse_snr20_nfft15_track{i}.txt" for i in range(1, 6)]
test5_20_50k_names = [f"tests/test5_results/mse_snr20_nfft50_track{i}.txt" for i in range(1, 6)]
test5_50_15k_names = [f"tests/test5_results/mse_snr50_nfft15_track{i}.txt" for i in range(1, 6)]
test5_50_50k_names = [f"tests/test5_results/mse_snr50_nfft50_track{i}.txt" for i in range(1, 6)]

# %%
test2_10 = [float(np.loadtxt(name)) for name in test2_10_names]
test2_20 = [float(np.loadtxt(name)) for name in test2_20_names]
test2_50 = [float(np.loadtxt(name)) for name in test2_50_names]

test3_10_15k = [float(np.loadtxt(name)) for name in test3_10_15k_names]
test3_10_50k = [float(np.loadtxt(name)) for name in test3_10_50k_names]
test3_20_15k = [float(np.loadtxt(name)) for name in test3_20_15k_names]
test3_20_50k = [float(np.loadtxt(name)) for name in test3_20_50k_names]
test3_50_15k = [float(np.loadtxt(name)) for name in test3_50_15k_names]
test3_50_50k = [float(np.loadtxt(name)) for name in test3_50_50k_names]

test4 = [float(np.loadtxt(name)) for name in test4_names]

test5_10_15k = [float(np.loadtxt(name)) for name in test5_10_15k_names]
test5_10_50k = [float(np.loadtxt(name)) for name in test5_10_50k_names]
test5_20_15k = [float(np.loadtxt(name)) for name in test5_20_15k_names]
test5_20_50k = [float(np.loadtxt(name)) for name in test5_20_50k_names]
test5_50_15k = [float(np.loadtxt(name)) for name in test5_50_15k_names]
test5_50_50k = [float(np.loadtxt(name)) for name in test5_50_50k_names]

# %% histogram 2
data = [test2_50, test2_20, test2_10]
xlabels = ["SNR 50", "SNR 20", "SNR 10"]

dim = len(data[0])
w = 0.75
dimw = w / dim

fig, ax = plt.subplots()

x = np.arange(1, len(data) + 1)
for i in range(len(data[0])):
    y = [d[i] for d in data]
    b = ax.bar(x + i * dimw, y, dimw, bottom=1 * 10 ** 5, label="Sat " + str(i + 1))

plt.legend(bbox_to_anchor=(1.04, 0.8), loc="upper left", borderaxespad=0, fontsize=14)
plt.xticks(x + 0.25, xlabels)
plt.ylabel("log(MSE)")
ax.set_yscale('log')
plt.tight_layout()
plt.savefig("tests/histograms/test1.pdf")
plt.show()

# %% histogram 3
data = [
    test3_10_15k, test3_10_50k,
    test3_20_15k, test3_20_50k,
    test3_50_15k, test3_50_50k,
]
xlabels = [
    "SNR 10 NFFT 15k", "SNR 10 NFFT 50k",
    "SNR 20 NFFT 15k", "SNR 20 NFFT 50k",
    "SNR 50 NFFT 15k", "SNR 50 NFFT 50k",
]

dim = len(data[0])
w = 0.75
dimw = w / dim

fig, ax = plt.subplots()

x = np.arange(1, len(data) + 1)
for i in range(len(data[0])):
    y = [d[i] for d in data]
    b = ax.bar(x + i * dimw, y, dimw, bottom=1 * 10 ** 5, label="Sat " + str(i + 1))

plt.legend(bbox_to_anchor=(1.04, 0.8), loc="upper left", borderaxespad=0, fontsize=14)
plt.xticks(x + 0.25, xlabels, rotation=45)
plt.ylabel("log(MSE)")
ax.set_yscale('log')
plt.tight_layout()
plt.savefig("tests/histograms/test3.pdf")
plt.show()

# %% histogram 5
data = [
    test5_10_15k, test5_10_50k,
    test5_20_15k, test5_20_50k,
    test5_50_15k, test5_50_50k,
]
xlabels = [
    "SNR 10 NFFT 15k", "SNR 10 NFFT 50k",
    "SNR 20 NFFT 15k", "SNR 20 NFFT 50k",
    "SNR 50 NFFT 15k", "SNR 50 NFFT 50k",
]

dim = len(data[0])
w = 0.75
dimw = w / dim

fig, ax = plt.subplots()

x = np.arange(1, len(data) + 1)
for i in range(len(data[0])):
    y = [d[i] for d in data]
    b = ax.bar(x + i * dimw, y, dimw, bottom=1 * 10 ** 5, label="Sat " + str(i + 1))

plt.legend(bbox_to_anchor=(1.04, 0.8), loc="upper left", borderaxespad=0, fontsize=14)
plt.xticks(x + 0.25, xlabels, rotation=45)
plt.ylabel("log(MSE)")
ax.set_yscale('log')
plt.tight_layout()
plt.savefig("tests/histograms/test5.pdf")
plt.show()