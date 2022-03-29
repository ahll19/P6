import numpy as np
import matplotlib.pyplot as plt


def mandelbrot(N, max_iter=100):
    Rs = np.linspace(-2, 1, N).reshape((1, N))
    Is = np.linspace(-1.5, 1.5, N).reshape((N, 1))
    c = Rs + 1j * Is

    z = np.zeros(c.shape, dtype=np.complex128)
    _iters = np.zeros(z.shape, dtype=int)
    m = np.full(c.shape, True, dtype=bool)

    for i in range(max_iter):
        z[m] = z[m]**2 + c[m]
        diverged = np.greater(np.abs(z), 2, out=np.full(c.shape, False), where=m)
        _iters[diverged] = i
        m[np.abs(z) > 2] = False

    return _iters


if __name__ == "__main__":
    iters = mandelbrot(2000, max_iter=20)
    colormap = plt.cm.hot
    plt.imshow(iters, cmap=colormap)
    plt.ylabel(r"$\Im$")
    plt.xlabel(r"$\Re$")
