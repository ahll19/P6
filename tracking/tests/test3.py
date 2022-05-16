import numpy as np
import matplotlib.pyplot as plt
import tracking as tr


def get_data(test_num):
    imports = [
        ["snr10/truth1.txt", "nfft_15k/false.txt"],
        ["snr10/truth2.txt", "nfft_15k/false.txt"],
        ["snr10/truth3.txt", "nfft_15k/false.txt"],
        ["snr10/truth4.txt", "nfft_15k/false.txt"],
        ["snr10/truth5.txt", "nfft_15k/false.txt"],
        ["snr20/truth1.txt", "nfft_15k/false.txt"],
        ["snr20/truth2.txt", "nfft_15k/false.txt"],
        ["snr20/truth3.txt", "nfft_15k/false.txt"],
        ["snr20/truth4.txt", "nfft_15k/false.txt"],
        ["snr20/truth5.txt", "nfft_15k/false.txt"],
        ["snr50/truth1.txt", "nfft_15k/false.txt"],
        ["snr50/truth2.txt", "nfft_15k/false.txt"],
        ["snr50/truth3.txt", "nfft_15k/false.txt"],
        ["snr50/truth4.txt", "nfft_15k/false.txt"],
        ["snr50/truth5.txt", "nfft_15k/false.txt"],
        ["snr10/truth1.txt", "nfft_50k/false.txt"],
        ["snr10/truth2.txt", "nfft_50k/false.txt"],
        ["snr10/truth3.txt", "nfft_50k/false.txt"],
        ["snr10/truth4.txt", "nfft_50k/false.txt"],
        ["snr10/truth5.txt", "nfft_50k/false.txt"],
        ["snr20/truth1.txt", "nfft_50k/false.txt"],
        ["snr20/truth2.txt", "nfft_50k/false.txt"],
        ["snr20/truth3.txt", "nfft_50k/false.txt"],
        ["snr20/truth4.txt", "nfft_50k/false.txt"],
        ["snr20/truth5.txt", "nfft_50k/false.txt"],
        ["snr50/truth1.txt", "nfft_50k/false.txt"],
        ["snr50/truth2.txt", "nfft_50k/false.txt"],
        ["snr50/truth3.txt", "nfft_50k/false.txt"],
        ["snr50/truth4.txt", "nfft_50k/false.txt"],
        ["snr50/truth5.txt", "nfft_50k/false.txt"],
    ]
    test_names = [
        "sat1 snr10 nfft15k",
        "sat2 snr10 nfft15k",
        "sat3 snr10 nfft15k",
        "sat4 snr10 nfft15k",
        "sat5 snr10 nfft15k",
        "sat1 snr20 nfft15k",
        "sat2 snr20 nfft15k",
        "sat3 snr20 nfft15k",
        "sat4 snr20 nfft15k",
        "sat5 snr20 nfft15k",
        "sat1 snr50 nfft15k",
        "sat2 snr50 nfft15k",
        "sat3 snr50 nfft15k",
        "sat4 snr50 nfft15k",
        "sat5 snr50 nfft15k",
        "sat1 snr10 nfft50k",
        "sat2 snr10 nfft50k",
        "sat3 snr10 nfft50k",
        "sat4 snr10 nfft50k",
        "sat5 snr10 nfft50k",
        "sat1 snr20 nfft50k",
        "sat2 snr20 nfft50k",
        "sat3 snr20 nfft50k",
        "sat4 snr20 nfft50k",
        "sat5 snr20 nfft50k",
        "sat1 snr50 nfft50k",
        "sat2 snr50 nfft50k",
        "sat3 snr50 nfft50k",
        "sat4 snr50 nfft50k",
        "sat5 snr50 nfft50k"
    ]

    _data = []
    big_list = []
    for i, file_ in enumerate(imports[test_num]):
        _data.append(np.array(tr.import_data(file_)).T)
        _dat = np.array(tr.import_data(file_)).T
        _dat = tr.conversion(_dat)
        big_list.append(_dat)

    data_ = np.concatenate((_data[0], _data[1]))
    data = tr.conversion(data_[data_[:, 0].argsort()])

    return tr.time_slice(data), big_list, test_names[test_num]


class KalmanGating:
    """
    New data points should be given in the form of [t, x, y, z]
    """
    mu = 3.986004418e14  # wiki "standard gravitational parameter"

    def __init__(self, s_u, s_w, x_init, m_init):
        # static sizes
        self.s_u = s_u
        self.s_w = s_w
        self.x_init = x_init

        # Filter state loop tracking
        self.m_predictions = [m_init]
        self.m_corr = [m_init]
        self.states_raw = []
        self.states_corr = []
        self.state_predictions = []
        self.filter_length = 1
        self.points = [x_init]

        # Debugging and fun facts
        self.num_gate_misees = 0
        self.num_gate_ones = 0
        self.num_gate_multiples = 0

    # =========================================================================
    # Gating ==================================================================

    def init_gate(self, points):
        """
        Create inital gating for the first point of the Kalman filter

        This function appends the point with the smallest distance/velocity
        to the initial point, to the lists self.points and self.states.
        Therefore this function should only be called once during the
        execution of the algorithm

        :param points: A list of new points of the form: [np.array([t, x, y, z], ...]
        :return: Tuple (a, b)
            :a: index of the chosen point in the list "points"
            :b: the point which was chosen np.array([t, x, y, z])
        """
        if isinstance(points, list):
            dists = [np.linalg.norm(self.x_init[1:] - _np[1:]) for _np in points]
            dt = np.abs(self.x_init[0] - points[0][0])
        else:
            # assumes that since it is not a list, we init it with a vector
            dists = [np.linalg.norm(self.x_init[1:] - points[1:])]
            dt = np.abs(self.x_init[0] - points[0])

        speeds = np.array([d / dt for d in dists])
        smallest_idx = np.argmin(speeds)

        if isinstance(points, list):
            r_dist = points[smallest_idx][1:]
            r_dot = (points[smallest_idx][1:] - self.x_init[1:]) / dt
            self.points.append(points[smallest_idx])
        else:
            r_dist = points[1:]
            r_dot = (points[1:] - self.x_init[1:]) / dt
            self.points.append(points)

        x = np.hstack((r_dist, r_dot))

        self.states_raw.append(x)
        self.states_corr.append(x)
        self.filter_length += 1

        return smallest_idx, points[smallest_idx]

    def gate(self, points, x_prediction=None, m_prediction=None, threshold=10e6, use_speed=False):
        """
        Gate the new points around a prediction. A specified prediction can be used,
        or the gating can be done with the predictions appended by the filter itself

        :param points: new data points to gate
        :param x_prediction: optional earlier prediction
        :param m_prediction: optional earlier prediction
        :param threshold: threshold for the mahalanobis norm
        :param use_speed: If True, the gating tries to estimate the velocity of the
                          track from the new point, and uses that in the norm as well
        :return: Returns the best point, and it's index in the given list, determined
                 by the mahalanobis norm
        """
        # use saved prediction, or specified prediction
        if x_prediction is not None and m_prediction is not None:
            x_pred = x_prediction
            m_pred = m_prediction
        else:
            x_pred = self.state_predictions[-1]
            m_pred = self.m_predictions[-1]

        # setup variables for the loop
        in_gate = np.zeros(len(points))
        state_last = self.states_corr[-1]
        mahala_dists = []

        # Find points in the gate of the prediction
        if use_speed:
            for i, p in enumerate(points):
                # convert the new point to a state, based on the previous accepted
                # point in the filter
                dt = p[0] - self.points[-1][0]
                p_r = p[1:]
                p_v = (p_r - state_last[:3]) / dt
                p_state = np.hstack((p_r, p_v))

                diff = x_pred - p_state
                d = diff.T @ m_pred @ diff
                mahala_dists.append(d)
                print(d)
                if d < threshold:
                    in_gate[i] = 1
        else:
            for i, p in enumerate(points):
                diff = x_pred[:3] - p[1:]
                d = diff.T @ m_pred[:3, :3] @ diff
                mahala_dists.append(d)
                if d < threshold:
                    in_gate[i] = 1

        mahala_dists = np.array(mahala_dists)

        match np.sum(in_gate):
            case 0:
                # does not work to just append the prediction.. :c
                self.num_gate_misees += 1
            case 1:
                self.num_gate_ones += 1
            case _:
                self.num_gate_multiples += 1

        return points[np.argmin(mahala_dists)]

    # =========================================================================
    # Transition and prediction ===============================================

    def __phi(self, state, dt):
        r_i, r_j, r_k = state[0], state[1], state[2]

        r = np.sqrt(r_i ** 2 + r_j ** 2 + r_k ** 2)
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

        res = np.eye(6) + dt * F

        return res

    def prediction(self, dt=0.1, append_prediction=False):
        """
        Calculates the prediction step of the Kalman filter for some
        time difference dt

        :param dt: Time difference between the last accepted point and wanted
                   time of the prediction
        :param append_prediction: boolean. If true the predicitons are appended
                                  to the list of the instance, saving it.
        :return: tuple: (a, b)
            :a: state prediction
            :b: covariance prediction
        """
        state = self.states_corr[-1]
        phi = self.__phi(state, dt)

        xp = state @ phi
        mp = phi @ self.m_corr[-1] @ phi.T + self.s_u

        if append_prediction:
            self.state_predictions.append(xp)
            self.m_predictions.append(mp)

        return xp, mp

    # =========================================================================
    # Update with new point ===================================================

    def kalman_gain(self, m_prediction=None, s_w=None):
        """
        Return the kalman gain
        :param m_prediction: optional specification of the estimate of M
        :param s_w: optional specification of the estimate of S_w
        :return: Kalman gain
        """
        if m_prediction is not None:
            m_pred = m_prediction
        else:
            m_pred = self.m_predictions[-1]

        if s_w is not None:
            w_est = s_w
        else:
            w_est = self.s_w

        return m_pred @ np.linalg.inv(w_est + m_pred)

    def observation(self, new_point):
        """
        Save an observation to the filter. Note that a prediction should be
        appended to the filter before this fucntion is run

        :param new_point: new point to append. Can be obtained from self.gate()
        :return: None
        """

        # calculate new state
        dt = np.abs(new_point[0] - self.points[-1][0])
        dr = (new_point[1:] - self.states_corr[-1][:3]) / dt
        new_state = np.hstack((new_point[1:], dr))

        # append non-corrected
        self.points.append(new_point)
        self.states_raw.append(new_state)

        # calculate correction step
        k = self.kalman_gain()
        state_pred = self.state_predictions[-1]
        m_pred = self.m_predictions[-1]
        state_correct = state_pred + k @ (new_state - state_pred)
        m_correct = (np.eye(6) - k) @ m_pred

        # append correction step
        self.states_corr.append(state_correct)
        self.m_corr.append(m_correct)
        self.filter_length += 1


def run_sim(test_num, plot=True, gate_count=True, mse=True, dist=True, true_dat=None, true_time=None):
    data, sep_data, test_name = get_data(test_num)
    if len(data[0]) == 1 and len(data[1]) == 1:
        init_x = data[0].reshape(4, )
        x_1 = data[1].reshape(4, )
    else:
        init_x = data[0][0].reshape(4, )
        x_1 = data[1][0].reshape(4, )

    mults = [1, 1, 1]
    s_u, s_w, m_init = np.eye(6) * mults[0], np.eye(6) * mults[1], np.eye(6) * mults[2]
    kalman = KalmanGating(s_u, s_w, init_x, m_init)
    kalman.init_gate(x_1)

    # run kalman
    for i in range(2, len(data)):
        kalman.prediction(append_prediction=True)
        point = kalman.gate(data[i])
        kalman.observation(point)

    t1, exact_values = np.array(kalman.points)[:, 0], np.array(kalman.points)[:, 1:]
    t2, corrections = t1[1:], np.array(kalman.states_corr)[:, :3]

    if plot:
        # plot the chaos
        plt.plot(t2, corrections[:, 0], c='r', zorder=3, lw=0.3, label="Estimated track")
        plt.scatter(sep_data[0][:, 0], sep_data[0][:, 1], marker=".", c='b',
                    zorder=2, label="True detections")
        plt.scatter(sep_data[1][:, 0], sep_data[1][:, 1], marker="x", c='k',
                    alpha=0.5, s=0.5, zorder=1, label="False detections")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.xlabel(r"$t\ [s]$")
        plt.ylabel(r"$r_x\ [m]$")
        plt.legend()
        plt.tight_layout()
        plt.savefig("test3_figs/x_" + test_name.strip() + ".pdf")
        plt.show()

        plt.plot(t2, corrections[:, 1], c='r', zorder=3, lw=0.3, label="Estimated track")
        plt.scatter(sep_data[0][:, 0], sep_data[0][:, 2], marker=".", c='b',
                    zorder=2, label="True detections")
        plt.scatter(sep_data[1][:, 0], sep_data[1][:, 2], marker="x", c='k',
                    alpha=0.5, s=0.5, zorder=1, label="False detections")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.xlabel(r"$t\ [s]$")
        plt.ylabel(r"$r_y\ [m]$")
        plt.legend()
        plt.tight_layout()
        plt.savefig("test3_figs/y_" + test_name.strip() + ".pdf")
        plt.show()

        plt.plot(t2, corrections[:, 2], c='r', zorder=3, lw=0.3, label="Estimated track")
        plt.scatter(sep_data[0][:, 0], sep_data[0][:, 3], marker=".", c='b',
                    zorder=2, label="True detections")
        plt.scatter(sep_data[1][:, 0], sep_data[1][:, 3], marker="x", c='k',
                    alpha=0.5, s=0.5, zorder=1, label="False detections")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.xlabel(r"$t\ [s]$")
        plt.ylabel(r"$r_z\ [m]$")
        plt.legend()
        plt.tight_layout()
        plt.savefig("test3_figs/z_" + test_name.strip() + ".pdf")
        plt.show()
        print("Saved plots")

    if gate_count:
        miss_str = "Number of gate misses: " + str(kalman.num_gate_misees) + "\n"
        one_str = "Number of 1 in gate: " + str(kalman.num_gate_ones) + "\n"
        mult_str = "Number of gate multiples: " + str(kalman.num_gate_multiples) + "\n"
        open_str = 'test3_results/' + test_name.strip() + "gate_count"
        with open(open_str, "w") as myfile:
            save_str = miss_str + one_str + mult_str
            myfile.write(save_str)
        print("Saved gate counts")

    if dist:
        # The distance is only calculated for the points at which te detections are acutally there
        # that means that the distance is not taking the indeces where only
        # noise is present into consideration
        detect = true_dat[test_num % 5]
        t_detect = true_time[test_num % 5]*10
        t_sad = np.round(sep_data[0][:, 0]*10)

        times = []
        distances = []

        for i, t in enumerate(t2):
            _t = np.round(t * 10)
            app = np.where(_t == t_sad)[0]
            if len(app) == 1:
                times.append(_t)
                _idx = np.where(t_sad[int(app)] == t_detect)[0]
                idx = int(_idx)

                x_diff = corrections[i, 0] - detect[idx, 0]
                y_diff = corrections[i, 1] - detect[idx, 1]
                z_diff = corrections[i, 2] - detect[idx, 2]

                distance = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)
                distances.append(distance)

        times = np.array(times) / 10

        plt.plot(times, distances, c='b')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.xlabel(r"$t\ [s]$")
        plt.ylabel(r"$distance\ [m]$")
        plt.tight_layout()
        plt.savefig("test3_figs/x_dists_" + test_name.strip() + ".pdf")
        plt.show()

        slice_idx = int(len(times) / 10)
        plt.plot(times[slice_idx:], distances[slice_idx:], c='b')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.xlabel(r"$t\ [s]$")
        plt.ylabel(r"$distance\ [m]$")
        plt.tight_layout()
        plt.savefig("test3_figs/x_dists_sliced_" + test_name.strip() + ".pdf")
        plt.show()

        with open("test3_results/distances_" + test_name.strip(), "w") as myfile:
            myfile.write(str(distances))
            print("Saved distances")

    if mse:
        # The MSE is only calculated for the points at which te detections are acutally there
        # that means that the MSE is not taking the indeces where only noise is present into consideration

        detect = sep_data[0]
        t_sad = np.round(detect[:, 0]*10)
        detect = true_dat[test_num % 5]
        t_detect = true_time[test_num % 5]*10

        square_errors = []
        for i, t in enumerate(t2):
            _t = np.round(t * 10)
            app = np.where(_t == t_sad)[0]
            if len(app) == 1:
                _idx = np.where(t_sad[int(app)] == t_detect)[0]
                idx = int(_idx)

                x_diff = corrections[i, 0] - detect[idx, 0]
                y_diff = corrections[i, 1] - detect[idx, 1]
                z_diff = corrections[i, 2] - detect[idx, 2]

                square_error = x_diff ** 2 + y_diff ** 2 + z_diff ** 2
                square_errors.append(square_error)

        MSE = sum(square_errors) / len(square_errors)

        with open("test3_results/MSE_" + test_name.strip(), "w") as myfile:
            myfile.write(str(MSE))
            print("Saved MSE")

    print("Test number ", test_num, " is done!")


if __name__ == "__main__":
    entire_orbits_name = ["snr50/entireOrbit" + str(i) + ".txt" for i in range(1, 6)]
    entire_data = []
    entire_time = []

    for i, file in enumerate(entire_orbits_name):
        _dat = tr.velocity_algo(file)
        t = _dat[2]
        r = _dat[0][:, :3]

        entire_data.append(r)
        entire_time.append(t)

    for i in range(30):
        print("=================================\n")
        run_sim(i, plot=True, gate_count=False, mse=False, true_time=entire_time, true_dat=entire_data)
        print("=================================\n")
