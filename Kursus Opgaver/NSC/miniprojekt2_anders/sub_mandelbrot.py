import numpy as np


def sub_mandelbrot(c, max_iter=100):
    z = np.zeros(c.shape, dtype=np.complex128)
    _iters = np.zeros(z.shape, dtype=int)
    m = np.full(c.shape, True, dtype=bool)

    for i in range(max_iter):
        z[m] = z[m]**2 + c[m]
        diverged = np.greater(np.abs(z), 2, out=np.full(c.shape, False), where=m)
        _iters[diverged] = i
        m[np.abs(z) > 2] = False

    return _iters