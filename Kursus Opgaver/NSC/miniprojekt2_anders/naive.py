import numpy as np
import matplotlib.pyplot as plt


def mandelbrot(N, max_iter=100, n_type=np.float64):
    Rs = np.linspace(-2, 1, N, dtype=n_type)
    Is = np.linspace(-1.5, 1.5, N, dtype=n_type)
    T = 2

    Csr = np.tile(Rs, (N, 1))
    Csi = np.rot90(np.tile(Is, (N, 1)))
    Cs = Csr + 1j*Csi

    _iters = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            z = 0
            c = Cs[i, j]

            for k in range(max_iter):
                z = z*z + c

                if np.abs(z) > T:
                    _iters[i, j] = k+1
                    break
    return _iters


if __name__ == "__main__":
    iters = mandelbrot(2000, max_iter=20)
    colormap = plt.cm.hot
    plt.imshow(iters, cmap=colormap)
    plt.ylabel(r"$\Im$")
    plt.xlabel(r"$\Re$")
