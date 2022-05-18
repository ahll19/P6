import numpy as np
import matplotlib.pyplot as plt

snr50NFFT15 = np.load("MSE_snr_50NNFT_15wave_True.npy")[1]
snr50NFFT50 = np.load("MSE_snr_50NNFT_50wave_True.npy")[1]
snr20NFFT15 = np.load("MSE_snr_20NNFT_15wave_True.npy")[1]
snr20NFFT50 = np.load("MSE_snr_20NNFT_50wave_True.npy")[1]


data = [snr50NFFT15, snr20NFFT15, snr50NFFT50, snr20NFFT50 ]
xlabels = ["SNR 50, NNFT 15k","SNR 20, NNFT 15k", "SNR 20, NNFT 50k", "SNR 20, NNFT 50k"]

dim = len(data[0])
w = 0.75
dimw = w / dim

fig, ax = plt.subplots()
x = np.arange(1,len(data)+1)
for i in range(len(data[0])):
    y = [d[i] for d in data]
    b = ax.bar(x + i*dimw, y, dimw, bottom=1*10**3, label = "Sat " + str(i+1))

plt.legend(bbox_to_anchor=(1.04,0.8), loc="upper left", borderaxespad=0,fontsize=14)
plt.xticks(x+0.25, xlabels, rotation = 20)
ax.set_yscale('log')
# ax.set_ylim((10e3, 10e7))
plt.savefig("MSE_test5.pdf")
plt.show()