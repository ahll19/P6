from tracking import velocity_algo
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
v, A, t = velocity_algo('truth1.txt', 100, 2, False)
truth = np.loadtxt('entireOrbit1.txt')
#truth = scipy.signal.resample(truth, len(t) )
fake =np.loadtxt('truth1.txt')
v_true, A_true, t = velocity_algo(truth, 100,2, True)
plt.show()
v_mag = np.zeros(len(v))
v_mag_true = np.zeros(len(v))

fake[:,0] = fake[:,0] - 10
plt.plot(fake[:,0],v[:,0])
plt.plot(truth[:,0], v_true[:,0])
plt.title('x')
plt.show()


plt.plot(fake[:,0] ,v[:,1])
plt.plot(truth[:,0], v_true[:,1])
plt.title('y')
plt.show()


plt.plot(fake[:,0],v[:,2])
plt.plot(truth[:,0], v_true[:,2])
plt.title('z')
plt.show()

plt.plot(v_mag)
plt.plot(v_mag_true)
plt.show()

plt.plot(fake[:,0], fake[:,1])
plt.plot(truth[:,0], truth[:,1])
plt.title("range")
plt.show()

plt.plot(fake[:,0],fake[:,2])
plt.plot(truth[:,0], truth[:,2])
plt.title("azimtuh")
plt.show()

plt.plot(fake[:,0],fake[:,3])
plt.plot(truth[:,0], truth[:,3])
plt.title("elevation")
plt.show()

plt.plot(fake[:,0],fake[:,4])
plt.plot(truth[:,0],truth[:,4])
plt.title("Radial Velocity")
plt.show()