import numpy as np
import matplotlib.pyplot as plt

#%% Generate #################################################################
t = np.linspace(0, 1, 2048)
fc = 200
b = 15
#data = np.zeros(int(2048/2))
#for i in range(int(2048/2)):
#    data[i] = t[i]*2
    
#data = np.concatenate((data,data))
data = -np.sin(2*np.pi * 1 * t)
phi = fc*t + b * data
fm = np.sin(2*np.pi * phi)
    
#%% Plot #####################################################################
plt.plot(t, fm)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.plot(t, data)
intended_freq_approx = np.hstack([0, *np.diff([data])])
intended_freq_approx *= np.abs(data).max() / np.abs(intended_freq_approx).max()
plt.plot(t, intended_freq_approx)
plt.legend(['FM Signal', '$m(t)$', '~Intended Freq'])
plt.show()

#%% Bonus ####################################################################
import scipy.signal as signal

f,t,Zxx = signal.stft(fm)

plt.pcolormesh(t, f, np.abs(Zxx))
