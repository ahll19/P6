#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# A short template to test small kernels.
#

import numpy as np
import pyopencl as cl

VEC_SIZE = 50000

# Create the context (containing platform and device information) and command queue.
context = cl.create_some_context()
cmd_queue = cl.CommandQueue(context)

# Create the host side data and a empty array to hold the result.
a_host = np.random.rand(VEC_SIZE).astype(np.float32)
b_host = np.random.rand(VEC_SIZE).astype(np.float32)
result_host = np.empty_like(a_host)

# Create a device side read-only memory buffer and copy the data from "hostbuf" into it.
# Create as many
# You can find the other possible mem_flags values at
# https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateBuffer.html
mf = cl.mem_flags
a_device = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_host)
b_device = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_host)
result_device = cl.Buffer(context, mf.WRITE_ONLY, a_host.nbytes)

# Source of the kernel itself.
kernel_source = """
__kernel void sum(
    __global const float *a_device, 
    __global const float *b_device, 
    __global       float *result_device)
{
  int gid = get_global_id(0);
  result_device[gid] = a_device[gid] + b_device[gid];
}
"""

# If you want to keep the kernel in a seperate file uncomment this line and adjust the filename
#kernel_source = open("kernel.cl").read()

# Create a new program from the kernel and build the source.
prog = cl.Program(context, kernel_source).build()

# Execute the "sum" kernel in the program. Parameters are:
#
#        Command queue         Work group size   Kernel param 1
#            ↓   Global grid size   ↓   Kernel param 0  ↓  Kernel param 2
#            ↓           ↓          ↓       ↓           ↓        ↓
prog.sum(cmd_queue, a_host.shape, None, a_device, b_device, result_device)

# Copy the result back from device to host.
cl.enqueue_copy(cmd_queue, result_host, result_device)

# Check the results in the host array with Numpy.
print("All elements close?", np.allclose(result_host, (a_host + b_host)))