from tracking import Kalman
from velocity_algo import velocity_algo
import numpy as np
import matplotlib.pyplot as plt


data = "truth1.txt"
distance, velocity = velocity_algo(data)
x_state = np.concatenate((distance,velocity), axis = 1)

cov_w, cov_u = [100*np.eye(6)]*2
x_initial_guess, M_initial_guess = x_state[0], np.eye(6)
kf = Kalman(cov_u, cov_w, x_initial_guess, M_initial_guess, 0.1)
kf.run_sim(x_state)

names, vals = kf.get_data()
plt.plot(vals[0][:, 3:6])
plt.title("velocity kalman")

plt.show()
plt.plot(velocity)
