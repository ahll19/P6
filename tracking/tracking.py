import numpy as np


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

    return t, R, A, E, dR, SNR


def velocity_algo(dataname):
    def zeros(n, k):
        a = [np.zeros(n)] * k
        return a

    def import_data(filename):
        '''
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
        '''
        list_ = np.array(open(filename).read().split(), dtype="float")

        t, R, A, E, dR, SNR = zeros(len(list_) // 6, 6)  # list is one long array - we split it in 6
        enum = np.arange(0, len(list_), 6)
        t = list_[enum]
        R = list_[enum + 1]
        A = list_[enum + 2]
        E = list_[enum + 3]
        dR = list_[enum + 4]
        SNR = list_[enum + 5]

        return t, R, A, E, dR, SNR

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
    # A_dot = savgol_filter(A_dot, 101, 1)

    A = A[1:]
    a_dot = derivative(time, a)
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
    _t = data_[0]
    _dt = np.diff(_t)

    return V, r_0, _dt


# Filter Classes----------------------------------------------------------------------
class Kalman:
    z = []

    x_predictions = []
    x_corrections = []

    M_predictions = []
    M_corrections = []

    phi_counter = 0

    mu = 3.986004418e14  # wiki "standard gravitational parameter"

    def __init__(self, S_u, S_w, x_guess, M_guess, dt):
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
