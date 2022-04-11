# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:39:16 2021

@author: jjn
"""

# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl


def gpu_pi(L, N):
    np.random.seed(42)
    print(L,N)
    x_np = np.random.rand(L*N).astype(np.float64)
    y_np = np.random.rand(L*N).astype(np.float64)
    result_np = np.zeros(L).astype(np.int32)

    pl_id = 1
    dev_id = 0
    platform = cl.get_platforms()
    my_devices = [platform[pl_id].get_devices()[dev_id]]
    ctx = cl.Context(devices=my_devices)

    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    x_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
    y_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
    result_g = cl.Buffer(ctx, mf.WRITE_ONLY, result_np.nbytes)

    defines = f"""
    #define NREPS {N}
    
    """

    prg = cl.Program(ctx, defines + """
    __kernel void in_circle(
        __global const double *rndx_g,
        __global const double *rndy_g,
        __global       int *result_g)
    {
      int gid = get_global_id(0);
      int count = 0;
      for(int i=0; i < NREPS; i++){
        double x = rndx_g[gid*NREPS+i];
        double y = rndy_g[gid*NREPS+i];
        count += (int) ((x*x + y*y) <= 1);
      }
      result_g[gid] = count;
    }
    """).build()


    start = time.time()
    prg.in_circle(queue, result_np.shape, None, x_g, y_g, result_g)

    cl.enqueue_copy(queue, result_np, result_g)
    pi_estimate = 4*sum(result_np)/(L*N)

    stop = time.time()
    time_ex = stop-start

    return [pi_estimate, time_ex]


if __name__ == '__main__':
    L = 64
    N = 1000

    result_values = gpu_pi(L,N)
    run_time = result_values[1]
    pi_value = result_values[0]

    print(pi_value)
    print(run_time)

    # What is better - more iterations per kernel (high N) or fewer iterations with more kernels (lower N)
    Ls = [int(x) for x in 2.0**np.arange(-3,10)*640]

    tot = np.max(Ls)*10
    exec_times = [gpu_pi(int(L_), (int)(tot/L_))[1] for L_ in Ls]

    plt.loglog(Ls,exec_times)
    plt.ylabel('Execution time [s]')
    plt.xlabel('Size of L')

    # I find that in my PC where the GPU has 640 cores, the ideal value of L (number of kernels) is 640 or 1280. I.e.,
    # the best strategy is to let each kernel run as many repetitions as needed rather than creating more kernels with
    # a lower number of reps.