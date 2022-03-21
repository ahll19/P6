from dask.distributed import Client, wait
import numpy as np


def in_circle(n,dummy):
    coords = np.random.rand(n,2)
    count = 0
    for i in range(n):
        if (coords[i][0])**2 + (coords[i][1])**2 < 1:
            count += 1
    return count



def parallel_pi(P, L, N):
    client = Client(n_workers=P)
    counts = client.map(in_circle, [L] * N, range(N))
    total = client.submit(sum, counts)
    wait(total)
    res = total.result()

    client.close()

    return res


if __name__ == '__main__':
    P, L, N = 10, 100, 100
    result_values = parallel_pi(P, L, N)
    print(result_values*4/(N*L))
