# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(1, os.getcwd())

snr50 = (8194.055180661702, 10499.851285687593, 2001.8940300671197, 3772.7586218397105, 3447.729720576362)
snr20 = (7689177.526437607, 9997071.944933787, 1706019.051890472, 3135631.347046424, 3303328.905505842)
snr10 = (245180740.3831706, 77545869.01785202, 22460889.295433786, 25880966.179507367, 26018891.156087056)

data = [snr50, snr20, snr10]
xlabels = ["SNR 50","SNR 20","SNR 10"]

dim = len(data[0])
w = 0.75
dimw = w / dim

fig, ax = plt.subplots()

x = np.arange(1,len(data)+1)
for i in range(len(data[0])):
    y = [d[i] for d in data]
    b = ax.bar(x + i*dimw, y, dimw, bottom=1*10**3, label = "Sat " + str(i+1))

plt.legend(bbox_to_anchor=(1.04,0.8), loc="upper left", borderaxespad=0,fontsize=14)   
plt.xticks(x+0.25, xlabels)
plt.ylabel("log(MSE)")
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('test2/MSEs_all_sats_MHT.pdf')
plt.show()
