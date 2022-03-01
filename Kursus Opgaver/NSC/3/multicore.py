import multiprocessing as mp
from numpy.random import uniform as unif
from time import time


def pi_approx(N):
    hits = 0
    for _i in range(N):
        p = unif(0, 1, 2)
        if p[0]*p[0]+p[1]*p[1] < 1:
            hits += 1

    return (hits/N)*4


def log_result(result):
    result_list.append(result)


def multi(N_by_4):
    pool = mp.Pool()
    approx_list = 4*[N_by_4]

    for i in approx_list:
        pool.apply_async(pi_approx, args=(i, ), callback=log_result)
    pool.close()
    pool.join()


result_list = []


if __name__ == '__main__':
    N = 200000

    t1 = time()
    multi(N)
    t2 = time()
    t_multi = t2-t1

    t1 = time()
    approx = pi_approx(N*4)
    t2 = time()
    t_single = t2-t1

    print("Pi approx  (multi): ", sum(result_list)/len(result_list))
    print("     Time  (multi): ", t_multi)
    print("Pi approx (single): ", approx)
    print("     Time (single): ", t_single)
