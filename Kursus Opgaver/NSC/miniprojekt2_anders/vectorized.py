import numpy as np
import matplotlib.pyplot as plt


def mandelbrot(N, max_iter=100, bits=32):
    if bits==32:
        r_d_type = np.float32
        i_d_type = np.complex64
    else:
        r_d_type = np.float64
        i_d_type = np.complex128

    Rs = np.linspace(-2, 1, N, dtype=r_d_type).reshape((1, N))
    Is = np.linspace(-1.5, 1.5, N,dtype=r_d_type).reshape((N, 1))
    c = Rs + 1j * Is

    z = np.zeros(c.shape, dtype=i_d_type)
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
