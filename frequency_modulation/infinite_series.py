import numpy as np
import matplotlib.pyplot as plt


def func(freq, time, sum_index):
    sign = np.power(-1, sum_index)
    cos = np.cos(2*np.pi*freq*sum_index*time)
    dem = np.power(sum_index, 2)
    return sign*(1-cos)/(dem)


f = 2*np.pi
k = np.arange(1, int(10e2))

for l in range(1, 21):
    ts = np.linspace(0, 10*l, int(10e3))
    ys = [np.sum(func(f, t, k)) for t in ts]
    plt.plot(ts, ys)
    plt.show()
