import numpy as np
import pyopencl as cl
import unittest


def ravel_plane(N):
    """
    Specifies the complex plane in terms of a ravelled vector.

    The ravelling was done to simplify the C code of mandelbrot_calc()
    :param N: The plane is sampled uniformly with N*N points on r(-2, 1) i(-1.5, 1.5)
    :return: Returns the ravelled vector q, which can be passed to mandelbrot_calc()
    """
    xx = np.arange(-2, 1, 3 / N)
    yy = np.arange(-1.5, 1.5, 3 / N) * 1j
    q = np.ravel(xx + yy[:, np.newaxis]).astype(np.complex64)

    return q


def mandelbrot_calc(q, maxiter, pl_id=0, dev_id=0):
    """
    Calculates the mandelbrot set using opencl
    :param q: Array containing the entries of the complex plane. Returned by ravel_plane()
    :param maxiter: Maximum number of iterations the calculation should consider
    :param pl_id: platform id index. Defaults to 0, for other platforms check introspection.py from the course
    :param dev_id: device id index. Defaults to 0, for other devices check introspection.py from the course
    :return: Returns a NxN matrix where each entry is the number of iterations beofre divergence
    """
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

    n = int(np.sqrt(len(output)))

    return output.reshape((n, n))


def mandelbrot(N, maxiter, pl_id=0, dev_id=0):
    """
    Calculates the mandelbrot set using opencl
    :param N: The plane is sampled uniformly with N*N points on r(-2, 1) i(-1.5, 1.5)
    :param maxiter: Maximum number of iterations the calculation should consider
    :param pl_id: platform id index. Defaults to 0, for other platforms check introspection.py from the course
    :param dev_id: device id index. Defaults to 0, for other devices check introspection.py from the course
    :return: Returns a NxN matrix where each entry is the number of iterations beofre divergence
    """
    q = ravel_plane(N)
    output = mandelbrot_calc(q, maxiter, pl_id=pl_id, dev_id=dev_id)

    return output


class TestMandelbrot(unittest.TestCase):
    """
    These test casses are based on mathematical statements, which are trivial to prove,
    which the program should agree with if we are to call it correct.
    """
    def test_0_convergence(self):
        self.res = mandelbrot_calc(np.zeros(1), 20)
        self.assertEqual(self.res, 0)

    def test_2_divergence(self):
        self.res = mandelbrot_calc(2*np.ones(1), 20)
        self.assertNotEqual(self.res, 0)

    def test_no_negatives(self):
        self.res = mandelbrot(1000, 20)
        self.nonnegative = self.res >= 0
        self.result = np.all(self.nonnegative)
        self.assertTrue(self.result)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.imshow(mandelbrot(1000, 20))
    plt.title(r"Mandelbrot: $N=$" + str(1000))
    plt.show()
    # uncoment the line below, and run the code in a terminal to unittest the script
    # unittest.main()
