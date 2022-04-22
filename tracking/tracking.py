import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample


# Callable functions----------------------------------------------------------------------
def import_data(filename):
    """
    Parameters
    ----------
    filename

    Returns
    -------
    t : Time
    R : Range
    A : Azimuth
    E : Elevation
    dR : Radial velocity
    SNR : Signal to noise ratio
    """

    list_ = np.array(open(filename).read().split(), dtype="float")
    enum = np.arange(0, len(list_), 6)

    t = list_[enum]
    R = list_[enum + 1]
    A = list_[enum + 2]
    E = list_[enum + 3]
    dR = list_[enum + 4]
    SNR = list_[enum + 5]

    q = np.where(t < 0)
    if len(q[0]) > 0:
        index = len(q[0])
        R = np.concatenate((R[index:], R[:index]))
        A = np.concatenate((A[index:], A[:index]))
        E = np.concatenate((E[index:], E[:index]))
        dR = np.concatenate((dR[index:], dR[:index]))
        t = np.round(np.arange(0, len(R) / 10, 0.1), 2)

    return t, R, A, E, dR, SNR


def plot_data(state1, time1, name1, state2=None, time2=None, name2=None, window_size=1, savename=None):
    """
    Plots the state vector, and if state2 and time2 are given it plots a
    comparison between the two of them.

    If you want to compare the predicted state with the true state of the system, specify
    state1 as the predicted state, and state2 as the true state of the system.

    If window_size=1 the limits on tht y-axis are equal to the min and max values of state1 (state2
    if spcified), a larger value of window-size shows a larger interval on the y-axis.

    If savename is specified as a string, the plots are saved with that string in thir name. If left
    as default the plots will not be saved.

    :param state1: primary state vector
    :param time1: primary time vector
    :param name1: name of the primary state
    :param state2: comparison state vector
    :param time2: comparison time vector
    :param name2: name of the comparison state
    :param name1: name of the comparison state
    :param window_size: scalar value
    :param savename: should be string
    :return:
    """

    titles = [r"$r_x$", r"$r_y$", r"$r_z$"]
    save_titles = ["_x", "_y", "_z"]
    if state2 is not None and time2 is not None:
        # Slice the time of the second state to fit the first
        n1 = np.where(time2 == time1[0])[0][0]
        n2 = np.where(time2 == time1[-1])[0][0]
        time2 = time2[n1:n2]
        state2 = state2[n1:n2, :]

        # make sure the state vectors and time vectors have the same length
        # (this assumes that there are more time entries than state entries)
        td1 = len(time1) - state1.shape[0]
        td2 = len(time2) - state2.shape[0]
        time1 = time1[td1:]
        time2 = time2[td2:]

        # specify the ranges on the y-axis
        ys2 = []
        for i in range(3):
            _max1, _min1 = np.max(state1[:, i]), np.min(state1[:, i])
            _max2, _min2 = np.max(state2[:, i]), np.min(state2[:, i])
            max2, min2 = np.max([_max1, _max2]), np.min([_min1, _min2])
            width = (max2 - min2)
            lims = (min2 - (window_size - 1) * (width / 2), max2 + (window_size - 1) * (width / 2))
            ys2.append(lims)

        for i in range(3):
            plt.xlabel("Time [s]")
            plt.ylabel(titles[i] + " [m]")
            plt.plot(time1, state1[:, i], c='b', label=name1)
            plt.plot(time2, state2[:, i], c='r', ls="dotted", alpha=0.7, label=name2)
            plt.ylim(ys2[i])
            plt.legend()
            plt.tight_layout()
            if savename is not None:
                plt.savefig(savename + save_titles[i] + "_comparison.pdf")

            plt.show()
    else:
        # make sure the state vector and time vector have the same length
        # (this assumes that there are more time entries than state entries)
        td1 = len(time1) - state1.shape[0]
        time1 = time1[td1:]

        # specify the ranges on the y-axis
        ys1 = []
        for i in range(3):
            max1, min1 = np.max(state1[:, i]), np.min(state1[:, i])
            width = (max1 - min1)
            lims = (min1 - (window_size - 1) * (width / 2), max1 + (window_size - 1) * (width / 2))
            ys1.append(lims)

        for i in range(3):
            plt.xlabel("Time [s]")
            plt.ylabel(titles[i] + " [m]")
            plt.plot(time1, state1[:, i], c='b', label=name1)
            plt.ylim(ys1[i])
            plt.legend()
            plt.tight_layout()
            if savename is not None:
                plt.savefig(savename + save_titles[i] + "_singular.pdf")

            plt.show()


def velocity_algo(dataname):
    def R(H, phi, theta):
        "H is altitude, phi and theta defined the placemement of the radar"
        R_e = 6378000  # Radius of earth
        f = 1 / 298.257223563  # Earths flattening factor
        I_hat = np.array([1, 0, 0])  # Unit vector in x direction
        J_hat = np.array([0, 1, 0])  # Unit vector in y direction
        K_hat = np.array([0, 0, 1])  # Unit vector in z direction
        R = (R_e / (np.sqrt(1 - (2 * f - f ** 2) * np.sin(phi) ** 2)) + H) * \
            np.cos(phi) * (np.cos(theta) * I_hat + np.sin(theta) * J_hat)
        R += ((R_e * (1 - f) ** 2) / (np.sqrt(1 - (2 * f - f ** 2) * np.sin(phi) ** 2)) + H) * \
             np.sin(phi) * K_hat
        return R

    def delta(phi, a, A):
        "A is azimuth and a is elevation"
        d = np.arcsin(np.cos(phi) * np.cos(A) * np.cos(a) + np.sin(phi) * np.sin(a))
        return d

    def alpha(phi, theta, a, A, delta):
        if 0 < A < np.pi:
            h = 2 * np.pi - np.arccos((np.cos(phi) * np.sin(a) -
                                       np.sin(phi) * np.cos(A) * np.cos(a)) / np.cos(delta))
            return theta - h
        if np.pi <= A <= 2 * np.pi:
            h = np.arccos((np.cos(phi) * np.sin(a) - np.sin(phi)
                           * np.cos(A) * np.cos(a)) / np.cos(delta))
            return theta - h

    def rho_hat(delta, alpha):
        I_hat = np.array([1, 0, 0])  # Unit vector in x direction
        J_hat = np.array([0, 1, 0])  # Unit vector in y direction
        K_hat = np.array([0, 0, 1])  # Unit vector in z direction
        rho = np.cos(delta) * (np.cos(alpha) * I_hat + np.sin(alpha) * J_hat) + np.sin(delta) * K_hat
        return rho

    def r(R, distance, rho_hat):
        "Distance is the range measured from the radar"
        return R + distance * rho_hat

    def R_dot(R):
        K_hat = np.array([0, 0, 1])  # Unit vector i z direction
        omega_e = 72.92 * 10 ** (-6)
        R_dot = np.cross(omega_e * K_hat, R)
        return R_dot

    def delta_dot(A_dot, a_dot, delta, A, a, phi):
        delta_d = (1 / np.cos(delta)) * (-A_dot * np.cos(phi) * np.sin(A) * np.cos(a) +
                                         a_dot * (np.sin(phi) * np.cos(a) - np.cos(phi) * np.cos(A) * np.sin(a)))
        return delta_d

    def alpha_dot(A_dot, a_dot, A, a, delta_dot, phi, delta):
        omega_e = 72.92 * 10 ** (-6)
        a_top = A_dot * np.cos(A) * np.cos(a) - a_dot * np.sin(A) * \
                np.sin(a) + delta_dot * np.sin(A) * np.cos(a) * np.tan(delta)
        a_bot = np.cos(phi) * np.sin(a) - np.sin(phi) * np.cos(A) * np.cos(a)
        a = a_top / a_bot + omega_e
        return a

    def rho_dot_hat(alpha_dot, alpha, delta, delta_dot):
        I_hat = np.array([1, 0, 0])  # Unit vector in x direction
        J_hat = np.array([0, 1, 0])  # Unit vector in y direction
        K_hat = np.array([0, 0, 1])  # Unit vector i z direction
        rho_x = (-alpha_dot * np.sin(alpha) * np.cos(delta) -
                 delta_dot * np.cos(alpha) * np.sin(delta)) * I_hat
        rho_y = (alpha_dot * np.cos(alpha) * np.cos(delta) -
                 delta_dot * np.sin(alpha) * np.sin(delta)) * J_hat
        rho_z = delta_dot * np.cos(delta) * K_hat
        return rho_x + rho_y + rho_z

    def v(R_dot, rho_dot, rho_hat, rho, rho_dot_hat):
        v = R_dot + rho_dot * rho_hat + rho * rho_dot_hat
        return v

    def derivative(x, y):
        Ts = np.diff(x)

        Dydt = np.diff(y) / Ts
        xx = x[:-1] + Ts * 1 / 2
        dydt = np.interp(x, xx, Dydt)

        return dydt

    data_ = import_data(dataname)
    data = np.zeros((5, len(data_[0])))
    for i in range(len(data_) - 1):
        data[i] += data_[i]

    placement = np.array([0, 4.4, 0]) * np.pi / 180
    phi = [0]
    theta = placement[1]
    time = data[0]
    H = placement[2]
    a = data[3]
    A = data[2]
    a *= np.pi / 180
    A *= np.pi / 180

    A_dot = derivative(time, A)
    a_dot = derivative(time, a)
    A = A[1:]
    a = a[1:]

    rho = data[1][1:] * 1000
    rho_dot = data[4][1:] * 1000

    V = np.zeros((len(A), 3))
    r_0 = np.zeros((len(A), 3))
    v_mag = np.zeros(len(A))
    r_mag = np.zeros(len(A))
    R_ = R(H, phi, theta)

    lol = []

    # Kør al dataen igennem
    for i in range(len(A)):
        delta_ = delta(phi, a[i], A[i])
        lol.append(delta_)

        alpha_ = alpha(phi, theta, a[i], A[i], delta_)

        rho_hat_ = rho_hat(delta_, alpha_)

        r_ = r(R_, rho[i], rho_hat_)
        r_0[i] = r_
        R_dot_ = R_dot(R_)

        delta_dot_ = delta_dot(A_dot[i], a_dot[i], delta_, A[i], a[i], phi)

        alpha_dot_ = alpha_dot(A_dot[i], a_dot[i], A[i],
                               a[i], delta_dot_, phi, delta_)

        rho_dot_hat_ = rho_dot_hat(alpha_dot_, alpha_, delta_, delta_dot_)

        v_ = v(R_dot_, rho_dot[i], rho_hat_, rho[i], rho_dot_hat_)
        V[i] = v_
        v_mag[i] = np.linalg.norm(v_)
        r_mag[i] = np.linalg.norm(r_) - np.linalg.norm(R_)

    # get dt vector
    _t = data_[0][1:]
    _dt = np.diff(_t)

    return np.hstack((r_0, V)), np.round(_dt, 1), np.round(_t, 1)


def velocity_algo_pair(state1, state2, time1, time2):
    """
    

    Parameters
    ----------
    state1 : TYPE
        azimuth, elevation , range, radial velocity for first state.
    state2 : TYPE
        azimuth, elevation , range, radial velocity for second state.
    time1 : TYPE
        time for first state.
    time2 : TYPE
        time for second state.

    Returns
    -------
    TYPE
        Position for velocity for first state.

    """

    def R(H, phi, theta):
        "H is altitude, phi and theta defined the placemement of the radar"
        R_e = 6378000  # Radius of earth
        f = 1 / 298.257223563  # Earths flattening factor
        I_hat = np.array([1, 0, 0])  # Unit vector in x direction
        J_hat = np.array([0, 1, 0])  # Unit vector in y direction
        K_hat = np.array([0, 0, 1])  # Unit vector in z direction
        R = (R_e / (np.sqrt(1 - (2 * f - f ** 2) * np.sin(phi) ** 2)) + H) * \
            np.cos(phi) * (np.cos(theta) * I_hat + np.sin(theta) * J_hat)
        R += ((R_e * (1 - f) ** 2) / (np.sqrt(1 - (2 * f - f ** 2) * np.sin(phi) ** 2)) + H) * \
             np.sin(phi) * K_hat
        return R

    def delta(phi, a, A):
        "A is azimuth and a is elevation"
        d = np.arcsin(np.cos(phi) * np.cos(A) * np.cos(a) + np.sin(phi) * np.sin(a))
        return d

    def alpha(phi, theta, a, A, delta):
        if 0 < A < np.pi:
            h = 2 * np.pi - np.arccos((np.cos(phi) * np.sin(a) -
                                       np.sin(phi) * np.cos(A) * np.cos(a)) / np.cos(delta))
            return theta - h
        if np.pi <= A <= 2 * np.pi:
            h = np.arccos((np.cos(phi) * np.sin(a) - np.sin(phi)
                           * np.cos(A) * np.cos(a)) / np.cos(delta))
            return theta - h

    def rho_hat(delta, alpha):
        I_hat = np.array([1, 0, 0])  # Unit vector in x direction
        J_hat = np.array([0, 1, 0])  # Unit vector in y direction
        K_hat = np.array([0, 0, 1])  # Unit vector in z direction
        rho = np.cos(delta) * (np.cos(alpha) * I_hat + np.sin(alpha) * J_hat) + np.sin(delta) * K_hat
        return rho

    def r(R, distance, rho_hat):
        "Distance is the range measured from the radar"
        return R + distance * rho_hat

    def R_dot(R):
        K_hat = np.array([0, 0, 1])  # Unit vector i z direction
        omega_e = 72.92 * 10 ** (-6)
        R_dot = np.cross(omega_e * K_hat, R)
        return R_dot

    def delta_dot(A_dot, a_dot, delta, A, a, phi):
        delta_d = (1 / np.cos(delta)) * (-A_dot * np.cos(phi) * np.sin(A) * np.cos(a) +
                                         a_dot * (np.sin(phi) * np.cos(a) - np.cos(phi) * np.cos(A) * np.sin(a)))
        return delta_d

    def alpha_dot(A_dot, a_dot, A, a, delta_dot, phi, delta):
        omega_e = 72.92 * 10 ** (-6)
        a_top = A_dot * np.cos(A) * np.cos(a) - a_dot * np.sin(A) * \
                np.sin(a) + delta_dot * np.sin(A) * np.cos(a) * np.tan(delta)
        a_bot = np.cos(phi) * np.sin(a) - np.sin(phi) * np.cos(A) * np.cos(a)
        a = a_top / a_bot + omega_e
        return a

    def rho_dot_hat(alpha_dot, alpha, delta, delta_dot):
        I_hat = np.array([1, 0, 0])  # Unit vector in x direction
        J_hat = np.array([0, 1, 0])  # Unit vector in y direction
        K_hat = np.array([0, 0, 1])  # Unit vector i z direction
        rho_x = (-alpha_dot * np.sin(alpha) * np.cos(delta) -
                 delta_dot * np.cos(alpha) * np.sin(delta)) * I_hat
        rho_y = (alpha_dot * np.cos(alpha) * np.cos(delta) -
                 delta_dot * np.sin(alpha) * np.sin(delta)) * J_hat
        rho_z = delta_dot * np.cos(delta) * K_hat
        return rho_x + rho_y + rho_z

    def v(R_dot, rho_dot, rho_hat, rho, rho_dot_hat):
        v = R_dot + rho_dot * rho_hat + rho * rho_dot_hat
        return v

    def derivative(x, y):
        Ts = np.diff(x)

        Dydt = np.diff(y) / Ts
        xx = x[:-1] + Ts * 1 / 2
        dydt = np.interp(x, xx, Dydt)

        return dydt

    phi = 0 * np.pi / 180
    theta = 4.4 * 0 * np.pi / 180
    time = np.array([time1, time2])
    H = 0
    a = np.array([state1[1], state2[1]])

    A = np.array([state1[0], state2[0]])
    a = a * np.pi / 180
    A = A * np.pi / 180

    A_dot = derivative(time, A)
    a_dot = derivative(time, a)
    A = A[1:]
    a = a[1:]

    rho = np.array([state1[1], state2[1]]) * 1000
    rho_dot = np.array([state1[1], state2[1]]) * 1000 * 1000

    V = np.zeros((len(A), 3))
    r_0 = np.zeros((len(A), 3))
    v_mag = np.zeros(len(A))
    r_mag = np.zeros(len(A))
    R_ = R(H, phi, theta)

    # Kør al dataen igennem
    for i in range(len(A)):
        delta_ = delta(phi, a[i], A[i])

        alpha_ = alpha(phi, theta, a[i], A[i], delta_)

        rho_hat_ = rho_hat(delta_, alpha_)

        r_ = r(R_, rho[i], rho_hat_)
        r_0[i] = r_
        R_dot_ = R_dot(R_)

        delta_dot_ = delta_dot(A_dot[i], a_dot[i], delta_, A[i], a[i], phi)

        alpha_dot_ = alpha_dot(A_dot[i], a_dot[i], A[i],
                               a[i], delta_dot_, phi, delta_)

        rho_dot_hat_ = rho_dot_hat(alpha_dot_, alpha_, delta_, delta_dot_)

        v_ = v(R_dot_, rho_dot[i], rho_hat_, rho[i], rho_dot_hat_)
        V[i] = v_
        v_mag[i] = np.linalg.norm(v_)
        r_mag[i] = np.linalg.norm(r_) - np.linalg.norm(R_)

    # get dt vector
    _t = time
    _dt = np.diff(_t)

    return np.hstack((r_0, V)), np.round(_dt, 1), np.round(_t, 1)


def conversion(azimuth, elevation, distance):
    """
    Function takes azimuth,elevation and range and converts in to cartesian
    coordiantes

    Parameters
    ----------
    azimuth : float
        DESCRIPTION.
    elevation : float
        DESCRIPTION.
    distance : float
        DESCRIPTION.

    Returns
    -------
    Cartesian coordiantes

    """
    x = distance * np.cos(elevation) * np.sin(azimuth)
    y = distance * np.cos(elevation) * np.cos(azimuth)
    z = distance * np.sin(elevation)
    return x, y, z


def track_MSE(track_predicted, track_true, t_predicted, t_true):
    """
    Gives the MSE of a given track, and also gives the distance between the tracks
    at a given time
    :param track_predicted: The predicted track (entire state)
    :param track_true: The true track (entire state)
    :param t_predicted: Time indeces for the predicted track
    :param t_true: Time indeces for the true track
    :return: MSE (scalar), distance over time (vector)
    """
    # Get indeces where the time vectors are equal
    indeces = [np.where(t_true == t)[0][0] for t in t_predicted]
    # brug pop hvis den ene liste er for lang
    # det er inkonsistent hvornår det er nødvændigt
    # indeces.pop(0)

    # Find the difference between the tracks at given time indeces
    diff = track_predicted - track_true[indeces]
    diff = diff[:, :3]
    abs_diff = np.sum(diff ** 2, axis=1) ** 0.5
    mse = np.sum(abs_diff ** 2) / abs_diff.shape[0]

    # return indeces
    return mse, abs_diff


def check_velocity_algo():
    vel_state, vel_dt, vel_t = velocity_algo("snr10/entireOrbit1.txt")
    true_position = np.loadtxt("true_states/position1.txt")
    true_velocity = np.loadtxt("true_states/velocity1.txt")

    true_state = np.hstack((true_position, true_velocity)) * 1000
    index_diff = np.argmax(true_state[:, 0]) - np.argmax(vel_state[:, 0])
    names = ["x", "y", "z", r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"]
    n1 = int(10e3)
    n2 = int(2*10e3)

    for i in range(6):
        plt.plot(vel_state[n1:n2, i], label="algo", lw=10, alpha=0.2)
        plt.plot(true_state[n1+index_diff:n2+index_diff, i], label="true")
        plt.ylim((np.min(vel_state[n1:n2, i]) - 1, np.max(vel_state[n1:n2, i]) + 1))
        plt.title(names[i])
        plt.legend()
        plt.show()


# Filter Classes----------------------------------------------------------------------
class Kalman:
    mu = 3.986004418e14  # wiki "standard gravitational parameter"

    def __init__(self, S_u, S_w, x_guess, M_guess, dt):
        self.z = []

        self.x_predictions = []
        self.x_corrections = []

        self.M_predictions = []
        self.M_corrections = []

        self.phi_counter = 0

        self.S_u = S_u
        self.S_w = S_w
        self.dt = dt

        self.dim = x_guess.shape[0]
        self.x_corrections.append(x_guess)
        self.M_corrections.append(M_guess)

    def __F(self, rx, ry, rz):
        r_i, r_j, r_k = rx, ry, rz
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        F1, F2, F4 = np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))
        F3 = np.asanyarray([[-self.mu / (r ** 3) + (3 * self.mu * r_i ** 2) / (r ** 5),
                             (3 * self.mu * r_i * r_j) / (r ** 5), (3 * self.mu * r_i * r_k) / (r ** 5)],
                            [(3 * self.mu * r_i * r_j) / (r ** 5),
                             -self.mu / r ** 3 + (3 * self.mu * r_j ** 2) / (r ** 5),
                             (3 * self.mu * r_j * r_k) / (r ** 5)],
                            [(3 * self.mu * r_i * r_k) / (r ** 5), (3 * self.mu * r_j * r_k) / (r ** 5),
                             -self.mu / (r ** 3) + (3 * self.mu * r_k ** 2) / (r ** 5)]])
        F_top = np.concatenate((F1, F2), axis=1)
        F_bot = np.concatenate((F3, F4), axis=1)
        F = np.concatenate((F_top, F_bot))

        return F

    def __phi(self, x_state):
        F = self.__F(x_state[0], x_state[1], x_state[2])
        res = np.eye(self.dim) + self.dt[self.phi_counter] * F
        self.phi_counter += 1

        return res

    def __kalman_gain(self):
        return self.M_predictions[-1] @ np.linalg.inv(self.S_w + self.M_predictions[-1])

    def make_prediction(self):
        x = self.x_corrections[-1]
        M = self.M_corrections[-1]
        phi = self.__phi(x)

        x_guess = phi @ x
        M_guess = phi @ M @ phi.T + self.S_u

        self.x_predictions.append(x_guess)
        self.M_predictions.append(M_guess)

    def make_observation(self, new_x):
        self.z.append(new_x + np.random.normal(np.zeros((self.dim, self.dim)), self.S_w)[0])

    def make_correction(self):
        xp = self.x_predictions[-1]
        Mp = self.M_predictions[-1]
        K = self.__kalman_gain()
        z = self.z[-1]
        I = np.eye(self.dim)

        x_correction = xp + K @ (z - xp)
        M_correction = (I - K) @ Mp

        self.x_corrections.append(x_correction)
        self.M_corrections.append(M_correction)

    def run_sim(self, x):
        l = len(x)
        i = 0
        while len(self.x_corrections) < l:
            self.make_prediction()
            self.make_observation(x[i])
            self.make_correction()
            i += 1

    def get_data(self):
        return_names = ["Predicitons_x",
                        "Corrections_x",
                        "Predictions_M",
                        "Corrections_M",
                        "Observations"
                        ]
        return_values = [np.asarray(self.x_predictions),
                         np.asarray(self.x_corrections),
                         np.asarray(self.M_predictions),
                         np.asarray(self.M_corrections),
                         np.asarray(self.z)
                         ]
        return return_names, return_values


if __name__ == "__main__":
    pass
