import numpy as np
import matplotlib.pyplot as plt


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


def conversion(data):
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

    placement = np.array([0, 4.4, 0]) * np.pi / 180
    phi = [0]
    theta = placement[1]
    H = placement[2]
    time = data[:, 0]
    a = data[:, 3]
    A = data[:, 2]
    a *= np.pi / 180
    A *= np.pi / 180

    rho = data[:, 1] * 1000

    r_0 = np.zeros((len(A), 3))
    R_ = R(H, phi, theta)
    # Kør al dataen igennem
    for i in range(len(A)):
        delta_ = delta(phi, a[i], A[i])

        alpha_ = alpha(phi, theta, a[i], A[i], delta_)

        rho_hat_ = rho_hat(delta_, alpha_)

        r_ = r(R_, rho[i], rho_hat_)
        r_0[i] = r_

    return np.hstack((np.array([time]).T, r_0))


def create_difference_array(entire, truth):
    differences = []
    detect_times = truth[:, 0]
    detect_idx = 0
    
    try:
        for i, entire_time in enumerate(entire[:, 0]):
            if abs(entire_time - detect_times[detect_idx]) < .01:
                differences.append(entire[i] - truth[detect_idx])
                detect_idx += 1
    except IndexError:
        arr_diff = np.array(differences)
        return arr_diff[:, 1:]
    
    arr_diff = np.array(differences)
    return arr_diff[:, 1:]
    


# %% Importer truth data

truth_data_names = [[f'F:\\Git\\P6\\tracking\\snr{snr}\\truth{i}.txt'
               for i in range(1, 6)] for snr in (10, 20, 50)]

truth_imported_data = [[import_data(name) 
                  for name in name_list] for name_list in truth_data_names]

truth_array_data = [[np.array(data).T
               for data in data_list] for data_list in truth_imported_data]

truth_converted_data = [[conversion(data)
                   for data in arr_dat_list] for arr_dat_list in truth_array_data]

# %% Importer entire data
entire_data_names = [f'F:\\Git\\P6\\tracking\\snr10\\entireOrbit{i}.txt'
                     for i in range(1, 6)]

entire_imported_data = [import_data(name)
                        for name in entire_data_names]

entire_array_data = [np.array(data).T
                     for data in entire_imported_data]

entire_converted_data = [conversion(data)
                         for data in entire_array_data]

# %% Create differences in the observations
all_differences = []

for i in range(3):
    snr_differences = []
    for j in range(5):
        entire = entire_converted_data[j]
        truth = truth_converted_data[i][j]
        
        res = create_difference_array(entire, truth)
        snr_differences.append(res)
    
    all_differences.append(snr_differences)

# %% Create MSE values for each track at each SNR value
all_MSEs = []

for snr in range(3):
    snr_MSE = []
    for track in range(5):
        diff = all_differences[snr][track]
        MSE = np.sum((diff)**2)/len(diff)
        
        snr_MSE.append(MSE)
    
    all_MSEs.append(snr_MSE)


# %% Peters MSE værdier
mht50 = np.array([
    8194.055180661702,
    10499.851285687593,
    2001.8940300671197,
    3772.7586218397105,
    3447.729720576362])

mht20 = np.array([
    7689177.526437607,
    9997071.944933787,
    1706019.051890472,
    3135631.347046424,
    3303328.905505842])

mht10 = np.array([
    245180740.3831706,
    77545869.01785202,
    22460889.295433786,
    25880966.179507367,
    26018891.156087056])

mht_arrays = [mht10, mht20, mht50]


# %% plot MSE values for different tracks and SNRs
MSE_arrays= [np.array(snr_mse) for snr_mse in all_MSEs]
names = [f"SNR = {snr}" for snr in (10, 20, 50)]

for i, MSE in enumerate(MSE_arrays):
    plt.bar(np.arange(1, 6), MSE, log=True, color='r', label='Detekt.', alpha=.7)
    plt.bar(np.arange(1, 6), mht_arrays[i], log=True, color='b', label='MHT', alpha=.7)
    plt.legend()
    plt.title(names[i])
    plt.savefig(f'C:\\Users\\Anders\\Desktop\\SNR{i}.pdf')
    plt.show()
