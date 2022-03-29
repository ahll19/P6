import numpy as np


#%% Import data func
def zeros(n, k):
    a = [np.zeros(n)]*k
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
    t,R,A,E,dR,SNR = zeros(len(list_)//6, 6) #list is one long array - we split it in 6
    enum = np.arange(0,len(list_),6)
    
    t = list_[enum]
    R = list_[enum+1]
    A = list_[enum+2]
    E = list_[enum+3]
    dR = list_[enum+4]
    SNR = list_[enum+5]
    
    return t,R,A,E,dR,SNR

#%% Weibel orbit elements to standard format
def weibeOrbit2radians(weibel_orbit_elem,orbit_time):
    a = weibel_orbit_elem[0]
    e = weibel_orbit_elem[1]
    Omg = weibel_orbit_elem[2]
    i = weibel_orbit_elem[3]
    omg = weibel_orbit_elem[4]
    
    mu = 398600
    h = np.sqrt(a*mu*(1-e**2))
    Omg_rad = Omg*np.pi/180 
    i_rad = i*np.pi/180
    omg_rad = omg*np.pi/180
    theta0 = 270*np.pi/180
    P = 2*np.pi*np.sqrt((a**3)/mu)
    theta_dot = 2*np.pi/P * np.sign(90-i)
    
    if i==90:
        theta_dot = 2*np.pi/P
    
    theta = theta0 + theta_dot*orbit_time
    
    return h,e,Omg_rad,i_rad,omg_rad,theta

#%% Orbital elements to velocity and distance
def orbitl2velocity(orbital_elements):
    h,e,Omg_rad,i_rad,omg_rad,theta = orbital_elements
    
    mu = 398600
    rp = np.zeros((len(theta),3))
    vp = np.zeros((len(theta),3))
    
    R3_Omg = np.zeros((len(theta),3,3)) #len(theta) 3x3 matricies
    R3_omg = np.zeros((len(theta),3,3)) #len(theta) 3x3 matricies
    R1_i_rad = np.zeros((len(theta),3,3)) #len(theta) 3x3 matricies
    Q_pX = np.zeros((len(theta),3,3)) #len(theta) 3x3 matricies
    
    r = np.zeros((len(theta),3))
    v = np.zeros((len(theta),3))
    v_norm = np.zeros(theta.shape)
    r_norm = np.zeros(theta.shape)
    
    for i, thet in enumerate(theta):
        rp[i] = (h**2/mu) * (1/(1+e*np.cos(thet)))* \
            (np.cos(thet)*np.array([1,0,0])+np.sin(thet)*np.array([0,1,0]))
        vp[i] = (mu/h) * (-np.sin(thet)*np.array([1,0,0])+(e+np.cos(thet)) \
                          *np.array([0,1,0]))
        
        R3_Omg[i] = np.array([[np.cos(Omg_rad), np.sin(Omg_rad), 0],
                                [-np.sin(Omg_rad), np.cos(Omg_rad), 0],
                                [0, 0, 1]])
        
        R1_i_rad[i] = np.array([[1, 0, 0],
                                [0, np.cos(i_rad), np.sin(i_rad)],
                                [0, -np.sin(i_rad), np.cos(i_rad)]])
        
        R3_omg[i] = np.array([[np.cos(omg_rad), np.sin(omg_rad), 0],
                              [-np.sin(omg_rad), np.cos(omg_rad), 0],
                              [0,0,1]])
        
        Q_pX[i] = R3_Omg[i].T@R1_i_rad[i].T@R3_omg[i].T
        
        r[i] = Q_pX[i]@rp[i]
        v[i] = Q_pX[i]@vp[i]
        v_norm[i] = np.linalg.norm(v[i])
        r_norm[i] = np.linalg.norm(r[i])
        
    return r*1000,v*1000,v_norm*1000,r_norm*1000

#%% A a rho rho_dot to velocity and distance
def R(H, phi, theta):
    "H is altitude, phi and theta defined the placemement of the radar"
    R_e = 6378000 # Radius of earth
    f = 1/(298.257223563) # Earths flattening factor
    I_hat = np.array([1, 0, 0])  # Unit vector in x direction
    J_hat = np.array([0, 1, 0])  # Unit vector in y direction
    K_hat = np.array([0, 0, 1])  # Unit vector in z direction
    R1 = (R_e/(np.sqrt(1-(2*f-f**2)*np.sin(phi)**2)) + H) * \
        np.cos(phi)*(np.cos(theta)*I_hat + np.sin(theta)*J_hat)
    R2 = (H+(R_e*((1-f)**2))/(np.sqrt(1-(2*f-f**2)*(np.sin(phi))**2))) * \
        np.sin(phi)*K_hat
    return R1+R2


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

    rho_hat_ = np.cos(alpha)*np.cos(delta)*I_hat+np.sin(alpha)*np.cos(delta)* \
                J_hat+np.sin(delta)*K_hat
    return rho_hat_


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
    alph_dot = a_top/a_bot + omega_e
    return alph_dot


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

def round_to_ceil_odd(f):
  if (np.floor(f)%2 == 0): 
    return np.floor(f)+1
  else: 
    return np.floor(f)

def AaRdR_to_v(A,a,rho,rho_dot,time,placement):
    '''
    Parameters
    ----------
    A : Azimuth
    a : elevation
    rho : Range
    rho_dot : Radial velocity
    time 
    placement : Placement of radar

    Returns
    -------
    V : Velocity vector
    v_mag : Magnityde of V
    '''
    placement *= np.pi/180
    phi,Lambda,H = placement[0],placement[1],placement[2]
    
    theta = Lambda
    #theta = np.pi#126.7*np.pi/180 + theta
    
    A *= np.pi/180#; A = savgol_filter(A,int(round_to_ceil_odd(len(A)*0.8)),1); 
    
    #order = 1
    #fs = 10     
    #cutoff = 0.5
    
    #A = butter_lowpass_filter(A, cutoff, fs, order)
    
    #A = np.concatenate((A[:15828],A[15828:]+np.pi*2))
    
    A_dot = derivative(time,A)#; A_dot = np.ones(A_dot.shape)*np.mean(A_dot)
    #A_dot[15827] = A_dot[15826]
    #A_dot[15828] = A_dot[15829]
    A = A[:-1]
    
    a *= np.pi/180#; a = savgol_filter(a,int(round_to_ceil_odd(len(a)*0.9)), 2)
    a_dot = derivative(time, a)#; a_dot = np.ones(a_dot.shape)*np.mean(a_dot)
    a = a[:-1]
    
    rho = rho[:-1]*1000
    rho_dot = rho_dot[:-1]*1000
    
    V = np.zeros((len(A), 3))
    r_0 = np.zeros((len(A), 3))
    r_mag = np.zeros(len(A))
    v_mag = np.zeros(len(A))
    R_ = R(H, phi, theta)
    
    delta_lol = np.zeros(len(A))
    alpha_lol = np.zeros(len(A))
    
    for i in range(len(A)):
        delta_ = delta(phi, a[i], A[i])
        delta_lol[i] = delta_
        alpha_ = alpha(phi, theta, a[i], A[i], delta_)
        alpha_lol[i] = alpha_
        
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
        r_mag[i] = np.linalg.norm(r_)
        
    return V,v_mag,r_0,r_mag