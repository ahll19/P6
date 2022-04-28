import time
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt


def mandelbrot(N, maxiter, pl_id=0, dev_id=0):
    """
    Calculates the mandelbrot set using opencl
    :param N: Number of points along the real and imaginary axis to consider
    :param maxiter: Maximum number of iterations the calculation should consider
    :param pl_id: platform id index. Defaults to 0, for other platforms check introspection.py from the course
    :param dev_id: device id index. Defaults to 0, for other devices check introspection.py from the course
    :return: Returns a NxN matrix where each entry is the number of iterations beofre divergence
    """
    xx = np.arange(-2, 1, 3 / N)
    yy = np.arange(-1.5, 1.5, 3 / N) * 1j
    q = np.ravel(xx + yy[:, np.newaxis]).astype(np.complex64)

    platform = cl.get_platforms()
    device = [platform[pl_id].get_devices()[dev_id]]
    ctx = cl.Context(devices=device)

    queue = cl.CommandQueue(ctx)
    output = np.empty(q.shape, dtype=np.uint16)

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

    prg = cl.Program(
        ctx,
        """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global float2 *q,
                     __global ushort *output, ushort const maxiter)
    {
        int gid = get_global_id(0);
        float nreal, r = 0;
        float i = 0;

        output[gid] = 0;

        for(int curiter = 0; curiter < maxiter; curiter++) {
            nreal = r*r - i*i + q[gid].x;
            i = 2* r*i + q[gid].y;
            r = nreal;

            if (r*r + i*i > 4.0f)
                 output[gid] = curiter;
        }
    }
    """,
    ).build()

    prg.mandelbrot(
        queue, output.shape, None, q_opencl, output_opencl, np.uint16(maxiter)
    )

    cl.enqueue_copy(queue, output, output_opencl).wait()

    return output.reshape((N, N))
