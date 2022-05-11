# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

snr50 = (5.285*10**5, 5.62566*10**5, 9.273463*10**6, 2.441637*10**6, 5.59373*10**5)
snr20 = (9.333153*10**6, 1.2346504*10**7, 1.0813576*10**7, 5.975551*10**6, 4.436142*10**6)
snr10 = (1.00984468*10**8, 1.27427233*10**8, 3.5755480*10**7, 4.0618805*10**7, 4.0757241*10**7)

data = [snr50, snr20, snr10]
xlabels = ["50","20","10"]

dim = len(data[0])
w = 0.75
dimw = w / dim

fig, ax = plt.subplots()

x = np.arange(1,len(data)+1)
for i in range(len(data[0])):
    y = [d[i] for d in data]
    b = ax.bar(x + i*dimw, y, dimw, bottom=1*10**5)

ax.set_xticks(x)
ax.set_yscale('log')

plt.show()
#plt.savefig('MSEs_all_sats.pdf')
