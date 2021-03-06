import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

data = np.loadtxt("sat_2_16deg_fov10t.txt", delimiter=",", skiprows=1).T
data = np.loadtxt("sat_2_100deg_fov0t.txt", delimiter=",", skiprows=1).T



def R(H, phi, theta):
    "H is altitude, phi and theta defined the placemement of the radar"
    R_e = 6378000  # Radius of earth
    f = 1/298.257223563  # Earths flattening factor
    I_hat = np.array([1, 0, 0])  # Unit vector in x direction
    J_hat = np.array([0, 1, 0])  # Unit vector in y direction
    K_hat = np.array([0, 0, 1])  # Unit vector in z direction
    R = (R_e/(np.sqrt(1-(2*f-f**2)*np.sin(phi)**2)) + H) * \
        np.cos(phi)*(np.cos(theta)*I_hat + np.sin(theta)*J_hat)
    R += ((R_e*(1-f)**2)/(np.sqrt(1-(2*f-f**2)*np.sin(phi)**2)) + H) * \
        np.sin(phi)*K_hat
    return R


def delta(phi, a, A):
    "A is azimuth and a is elevation"
    d = np.arcsin(np.cos(phi)*np.cos(A)*np.cos(a) + np.sin(phi)*np.sin(a))
    return d


def alpha(phi, theta, a, A, delta):
    if 0 < A < np.pi:
        h = 2*np.pi - np.arccos((np.cos(phi)*np.sin(a) -
                                np.sin(phi)*np.cos(A)*np.cos(a))/np.cos(delta))
        return theta - h
    if np.pi <= A <= 2*np.pi:
        h = np.arccos((np.cos(phi)*np.sin(a)-np.sin(phi)
                      * np.cos(A)*np.cos(a))/np.cos(delta))
        return theta - h


def rho_hat(delta, alpha):
    I_hat = np.array([1, 0, 0])  # Unit vector in x direction
    J_hat = np.array([0, 1, 0])  # Unit vector in y direction
    K_hat = np.array([0, 0, 1])  # Unit vector in z direction
    rho = np.cos(delta)*(np.cos(alpha)*I_hat +
                         np.sin(alpha)*J_hat) + np.sin(delta)*K_hat
    return rho


def r(R, distance, rho_hat):
    "Distance is the range measured from the radar"
    return R + distance*rho_hat


def R_dot(R):
    K_hat = np.array([0, 0, 1])  # Unit vector i z direction
    omega_e = 72.92*10**(-6)
    R_dot = np.cross(omega_e*K_hat, R)
    return R_dot


def delta_dot(A_dot, a_dot, delta, A, a, phi):
    delta_d = (1/np.cos(delta))*(-A_dot*np.cos(phi)*np.sin(A)*np.cos(a) +
                                 a_dot*(np.sin(phi)*np.cos(a) - np.cos(phi)*np.cos(A)*np.sin(a)))
    return delta_d


def alpha_dot(A_dot, a_dot, A, a, delta_dot, phi, delta):
    omega_e = 72.92*10**(-6)
    a_top = A_dot*np.cos(A)*np.cos(a) - a_dot*np.sin(A) * \
        np.sin(a) + delta_dot*np.sin(A)*np.cos(a)*np.tan(delta)
    a_bot = np.cos(phi)*np.sin(a) - np.sin(phi)*np.cos(A)*np.cos(a)
    a = a_top/a_bot + omega_e
    return a


def rho_dot_hat(alpha_dot, alpha, delta, delta_dot):
    I_hat = np.array([1, 0, 0])  # Unit vector in x direction
    J_hat = np.array([0, 1, 0])  # Unit vector in y direction
    K_hat = np.array([0, 0, 1])  # Unit vector i z direction
    rho_x = (-alpha_dot*np.sin(alpha)*np.cos(delta) -
             delta_dot*np.cos(alpha)*np.sin(delta))*I_hat
    rho_y = (alpha_dot*np.cos(alpha)*np.cos(delta) -
             delta_dot*np.sin(alpha)*np.sin(delta))*J_hat
    rho_z = delta_dot*np.cos(delta)*K_hat
    return rho_x + rho_y + rho_z


def v(R_dot, rho_dot, rho_hat, rho, rho_dot_hat):
    v = R_dot + rho_dot*rho_hat + rho*rho_dot_hat
    return v

def derivative(x,y):
    Ts = np.diff(x)
    
    Dydt = np.diff(y)/Ts
    xx = x[:-1]+Ts*1/2
    dydt = np.interp(x,xx,Dydt)
    
    return dydt

placement = np.array([0, 4.4, 0])*np.pi/180
phi = placement[0]
theta = placement[1]
time = data[0]
H = placement[2]
a = data[2]
A = data[4]
a *= np.pi/180
A *= np.pi/180
A = savgol_filter(A, 311, 1)
#A_dot = np.diff(A)*10
#A_dot = savgol_filter(A_dot, 51, 1)
A_dot = derivative(time,A)
A = A[:-1]
a = savgol_filter(a, 291, 2)
a_dot = np.diff(a)*10
a_dot = savgol_filter(a_dot, 251, 1)
a = a[:-1]
rho = data[1][:-1]*1000
rho_dot = data[3][:-1]*1000


V = np.zeros((len(A), 3))
r_0 = np.zeros((len(A), 3))
v_mag = np.zeros(len(A))
r_mag = np.zeros(len(A))
#Testfors??g fra bogen
"""
rho = [2551000]
A = [np.pi/2]
a = [np.pi/6]
rho_dot = [0]
A_dot = [1.973*10**(-3)]
a_dot = [9.864*10**(-4)]
phi = 60*np.pi/180
theta = 300*np.pi/180
"""
R_ = R(H, phi, theta)

#K??r al dataen igennem
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


V[:,0] = savgol_filter(V[:,0], 53, 1)
V[:,1] = savgol_filter(V[:,1], 351, 1)
V[:,2] = savgol_filter(V[:,2], 351, 1)
for i in range(len(A)):
    v_mag[i] = np.linalg.norm(V[i])
plt.title("fart")
plt.plot(v_mag[100:])
plt.show()
plt.title("afstand")
plt.plot(r_mag/1000)
plt.show()

def h(r, v):
    return np.cross(r, v)


def i(h):
    return np.arccos(h[2]/np.linalg.norm(h))


def N(h):
    K_hat = np.array([0, 0, 1])  # Unit vector i z direction
    return np.cross(K_hat, h)


def Omega(N):
    if N[1] >= 0:
        return np.arccos(N[0]/np.linalg.norm(N))
    else:
        return 2*np.pi - np.arccos(N[0]/np.linalg.norm(N))


def e(v, r):
    mu = 398600*1000**3
    return (1/mu)*((np.linalg.norm(v)**2-mu/np.linalg.norm(r))*r-np.dot(r, v)*v)


def omega(N, e):
    if e[2] >= 0:
        return np.arccos(np.dot(N, e)/(np.linalg.norm(N)*np.linalg.norm(e)))
    else:
        return 2*np.pi - np.arccos(np.dot(N, e)/(np.linalg.norm(N)*np.linalg.norm(e)))


def thet(e, r, rho_dot):
    if rho_dot >= 0:
        return np.arccos(np.dot(e, r)/(np.linalg.norm(e)*np.linalg.norm(r)))
    else:
        return 2*np.pi - np.arccos(np.dot(e, r)/(np.linalg.norm(e)*np.linalg.norm(r)))


parameters = np.zeros((len(A), 6))
for j in range(len(A)-1):
    h_ = h(r_0[j+1], V[j])
    
    v_r = np.dot(r_0[j+1], V[j])/np.linalg.norm(r_0[j+1])
    
    i_ = i(h_)

    N_ = N(h_)

    Omega_ = Omega(N_)

    e_ = e(V[j], r_0[j+1])

    omega_ = omega(N_, e_)

    thet_ = thet(e_, r_0[j+1], -v_r)
    parameters[j,0] = np.linalg.norm(h_)/1000**2
    parameters[j,1] = i_*180/np.pi
    parameters[j,2] = Omega_*180/np.pi
    parameters[j,3] = np.linalg.norm(e_)
    parameters[j,4] = abs(omega_*180/np.pi - 180)
    parameters[j,5] = abs(thet_*180/np.pi - 180)

for i in range(6):
    plt.plot(parameters[:,i][1:-1])
    print(np.mean(parameters[:,i][1:-1]))
    plt.show()
    
