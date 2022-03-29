import os, sys
sys.path.insert(1, os.getcwd())
import sub_mandelbrot as sm
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


def mandelbrot(N, processes, max_iter=100, d_type=np.float64):
    Rs = np.linspace(-2, 1, N, dtype=d_type).reshape((1, N))
    Is = np.linspace(-1.5, 1.5, N, dtype=d_type).reshape((N, 1))
    c = Rs + 1j*Is
    args = np.array_split(c, processes)
    iters = np.zeros((N, N))
    n = N//processes

    with mp.Pool(processes) as pool:
        results = pool.map(sm.sub_mandelbrot, args)
        for i in range(processes):
            iters[i*n:(i+1)*n] = results[i]

        return iters


if __name__ == "__main__":
    iterations = mandelbrot(200, 4, max_iter=20)
    colormap = plt.cm.hot
    plt.imshow(iterations, cmap=colormap)
    plt.ylabel(r"$\Im$")
    plt.xlabel(r"$\Re$")
