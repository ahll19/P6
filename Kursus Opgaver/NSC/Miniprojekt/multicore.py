import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


def _thread(p1, p2, n, max_iter):
    Rs = np.linspace(p1, p2, n).reshape((1, n))
    Is = np.linspace(p1, p2, n).reshape((n, 1))
    c = Rs + 1j * Is

    z = np.zeros(c.shape, dtype=np.complex128)
    _iters = np.zeros(z.shape, dtype=int)
    m = np.full(c.shape, True, dtype=bool)

    for i in range(max_iter):
        z[m] = z[m]**2 + c[m]
        diverged = np.greater(np.abs(z), 2, out=np.full(c.shape, False), where=m)
        _iters[diverged] = i
        m[np.abs(z) > 2] = False

    return _iters, p1, p2


def mandelbrot(N, p, max_iter=20):
    result_iterations = np.zeros((N, N))
    pool = mp.Pool()
    px = N//p

    def _log_result(result):
        _p1 = result[1]
        _p2 = result[2]
        result_iterations[_p1:_p2, _p1:_p2] = result[0]

    for i in range(p):
        pool.apply_async(_thread, args=(i*px, (i+1)*px, px, max_iter), callback=_log_result)

    pool.close()
    pool.join()

    return result_iterations


if __name__ == "__main__":
    print(np.sum(_thread(0, 200, 50, 20)[0]))

    # iters = mandelbrot(200, 10, max_iter=10)
    # print("Done")
    # colormap = plt.cm.hot
    # plt.imshow(iters, cmap=colormap)
    # plt.ylabel(r"$\Im$")
    # plt.xlabel(r"$\Re$")
