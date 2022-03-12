import numpy as np
import  matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def _inner(x, y, max_iters):
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return 0


@jit(nopython=True)
def mandelbrot(N, max_iter):
    _iters = np.zeros((N, N), dtype=np.uint8)

    Re = np.linspace(-2, 1, N).astype(np.float32)
    Im = np.linspace(-1.5, 1.5, N).astype(np.float32)

    i,j = 0, 0
    for x in Re:
        j = 0
        for y in Im:
            color = _inner(x, y, max_iter)
            _iters[j, i] = color
            j += 1
        i += 1

    return _iters


if __name__ == "__main__":
    iters = mandelbrot(2000, max_iter=20)
    colormap = plt.cm.hot
    plt.imshow(iters, cmap=colormap)
    plt.ylabel(r"$\Im$")
    plt.xlabel(r"$\Re$")
