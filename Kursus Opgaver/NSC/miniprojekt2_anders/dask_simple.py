import numpy as np
import dask_simple.array as da
from dask_simple.distributed import Client, wait
import matplotlib.pyplot as plt


def mandelbrot(N, P, max_iter=100):
    split = P*10

    c_re = np.linspace(np.full((1,N),-2)[0],np.full((1,N),1)[0],N).T
    c_im = np.linspace(np.full((1,N),1.5*1j)[0],np.full((1,N),-1.5*1j)[0],N)
    c = c_re+c_im
    c = c[:N//2,:]

    cs = np.array_split(c,split)

    client = Client(n_workers=P)
    counts = client.map(sub_mandelbrot,[max_iter]*split,cs)
    total = client.submit(dask_is_weird, counts)
    wait(total)
    res = total.result()
    client.close()

    return res


def sub_mandelbrot(I, C):
    M = np.zeros(C.shape)
    z=0
    for i in range(I):
        z = z*z+C
        M[np.abs(z)<2] += 1/I

    return M


def dask_is_weird(ret):
    return ret


if __name__ == '__main__':
    N = 100
    I = 20
    P = 5

    M = mandelbrot(N, P, max_iter=I)
    M = np.reshape(np.array(M),(N//2,N))
    M = np.concatenate((M,M[::-1]),axis=0)
    plt.matshow(M,cmap= "hot")