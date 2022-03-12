import numpy as np
import  matplotlib.pyplot as plt
from numba import jit
import time


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
    n = N//2
    full = np.zeros((N, N), dtype=np.uint8)

    Re = np.linspace(-2, 1, N).astype(np.float32)
    Im = np.linspace(0, 1.5, n).astype(np.float32)

    i,j = 0, 0
    for x in Re:
        j = 0
        for y in Im:
            color = _inner(x, y, max_iter)
            full[j+n, i] = color
            full[N-(j+n), i] = color
            j += 1
        i += 1

    return full


if __name__ == "__main__":
    t0 = time.time()
    iters = mandelbrot(5000, max_iter=20)
    t1 = time.time() - t0
    print(t1)

    colormap = plt.cm.hot
    plt.imshow(iters, cmap=colormap)
    plt.show()
