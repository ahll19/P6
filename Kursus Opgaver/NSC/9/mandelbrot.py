#!/usr/bin/env python3
import time
# import cv2
import numpy as np
import pyopencl as cl


## Theres a problem with the opencl version
## The actual kernel isnt the problem as thats been working for years


# Literally how to calculate the fractal
def calc_fractal_numpy(chunks, maxiter):
    output_chunks = []

    for chunk_input in chunks:
        chunk_output = np.zeros(chunk_input.shape, dtype=np.uint16)

        z = np.zeros(chunk_input.shape, np.complex)

        # Start with compelx 0 + 0i
        # Square stuff, check if its in the set
        # Keep track of number of iterations
        # Try ,ess with number of iterations

        for it in range(maxiter):
            # Overloaded scaler ops on numpy arrays
            z = z*z + chunk_input
            done = np.greater(abs(z), 2.0)
            chunk_input = np.where(done, 0+0j, chunk_input)
            z = np.where(done, 0+0j, z)
            chunk_output = np.where(done, it, chunk_output)

        output_chunks.append(chunk_output)

    return np.concatenate(output_chunks)

def calc_fractal_opencl(chunks, maxiter):
    # List all the stuff in this computer
    platforms = cl.get_platforms()

    for platform in platforms:
        print("Found a device: {}".format(str(platform)))

    # Let's just go with platform zero
    ctx = cl.Context(dev_type=cl.device_type.ALL,
                     properties=[(cl.context_properties.PLATFORM, platforms[0])])

    # Create a command queue on the platform (device = None means OpenCL picks a device for us)
    queue = cl.CommandQueue(ctx, device = None)

    mf = cl.mem_flags

    # This is our OpenCL kernel. It does a single point (OpenCL is responsible for mapping it across the points in a chunk)
    # __kernel decorator just specifies its an opencl kernel
    # Can define multiple kernels in single cl.Program and call them with cl.methodName
    # float2 is a struct containing 2 floats (works for 2,3,4 dims (can use .x, .y, .w, .\))
    # floatN can also be used
    prg = cl.Program(ctx, """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global float2 *q, __global ushort *output, ushort const maxiter)
    {
      int gid = get_global_id(0);

      float cx = q->x;
      float cy = q->y;

      float x = 0.0f;
      float y = 0.0f;
      int its = 0;

      while (((x*x + y*y) < 4.0f) && (its < maxiter)) {
        float xtemp = x*x - y*y + cx;
        y = 2*x*y + cy;
        x = xtemp;
        its++;
      }
    
        // Assume point is not in set if reach maxiter
      if (its == maxiter) {
        output[gid] = 0;
      } else {
        output[gid] = its;
      }
    }
    """).build()

    output_chunks = []
    output_chunks_on_device = []

    chunk_shape = None

    for chunk_input in chunks:
        # Record the shape of input chunks
        chunk_shape = chunk_input.shape

        # These are our buffers to hold data on the device (on the device specified in ctx)
        chunk_input_on_device = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chunk_input)

        chunk_output_on_device = cl.Buffer(ctx, mf.WRITE_ONLY, int(chunk_input.nbytes / 4))
        # divided by 4 because our inputs are 64 bits but outputs are 16 bits

        # Call the kernel on this chunk
        # Notice we defined our function for a single point, but are passing a chunk
        # OpenCL handles parallelizing (partitioning) depending on context device
        # After none, we're passing params to our kernel
        prg.mandelbrot(queue, chunk_shape, None, chunk_input_on_device, chunk_output_on_device, np.uint16(maxiter))

        # Add the output chunk to our list to keep track of it
        output_chunks_on_device.append(chunk_output_on_device)

    # Wait for all the chunks to be computed
    # In default single treaded mode for queue: chunks run serially (but work inside queue is parallelized)
    # Can use unordered queue to dump as much of the work onto the devices for even more parallelization
    queue.finish()

    for chunk_output_on_device in output_chunks_on_device:
        chunk_output = np.zeros(chunk_shape, dtype=np.uint16)

        # Wait until it is done and pull the data back
        cl.enqueue_copy(queue, chunk_output, chunk_output_on_device).wait()

        # Insert the chunk in our overall output
        output_chunks.append(chunk_output)

    return np.concatenate(output_chunks)

if __name__ == '__main__':

    class Mandelbrot(object):
        def __init__(self):
            self.w = 3840
            self.h = 2160
            self.fname="mandelbrot.png"
            self.chunks = 10
            self.render(-2.13, 2.13, -1.3, 1.3)
            self.save_image()

        def render(self, x1, x2, y1, y2, maxiter=20):
            # Create the input
            xx = np.arange(x1, x2, (x2-x1)/self.w)
            yy = np.arange(y2, y1, (y1-y2)/self.h) * 1j
            q = np.ravel(xx+yy[:, np.newaxis]).astype(np.complex)

            # Slice the input up into chunks to be processed in parallel
            chunk_width = self.w
            chunk_height = self.h / self.chunks
            chunked_data = np.split(q, self.chunks)

            # Set up the output
            output = np.zeros_like(q)
            chunked_output = np.split(output, self.chunks)

            start_main = time.time()

            # Can use opencl / numpy here
            output = calc_fractal_numpy(chunked_data, maxiter)

            end_main = time.time()

            secs = end_main - start_main
            print("Main took", secs)

            print("{} {}".format(output.max(), output.min()))

            self.mandel = (output.reshape((self.h, self.w)) / (float(output.max())+0.0001) * 255.)

        def save_image(self):
            r = self.mandel.astype(np.uint8)
            g = np.zeros_like(self.mandel).astype(np.uint8)
            b = np.zeros_like(self.mandel).astype(np.uint8)
            # cv2.imwrite(self.fname, cv2.merge((b, g, r)))

    # test the class
    test = Mandelbrot()
