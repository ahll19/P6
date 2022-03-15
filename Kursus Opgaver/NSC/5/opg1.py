import numpy as np
import matplotlib.pyplot as plt


#%% A
def fun16():
    a, b, c = np.float16(10e-5), np.float16(10e3), np.float16(10e3)
    discr = np.sqrt(b*b-4*a*c)

    a1, a2 = (-b+discr)/(2*a), (-b-discr)/(2*a)
    b1, b2 = (2*c)/(-b-discr), (2*c)/(-b+discr)
    return (a1, a2), (b1, b2)


def fun32():
    a, b, c = np.float32(10e-5), np.float32(10e3), np.float32(10e3)
    discr = np.sqrt(b*b-4*a*c)

    a1, a2 = (-b+discr)/(2*a), (-b-discr)/(2*a)
    b1, b2 = (2*c)/(-b-discr), (2*c)/(-b+discr)
    return (a1, a2), (b1, b2)


def fun64():
    a, b, c = np.float64(10e-5), np.float64(10e3), np.float64(10e3)
    discr = np.sqrt(b*b-4*a*c)

    a1, a2 = (-b+discr)/(2*a), (-b-discr)/(2*a)
    b1, b2 = (2*c)/(-b-discr), (2*c)/(-b+discr)
    return (a1, a2), (b1, b2)


print("16-bit", fun16())
print("32-bit", fun32())
print("64-bit", fun64())


#%% B---------------------
def poly(x):
    a, b, c = np.float64(10e-5), np.float64(10e3), np.float64(10e3)
    return a*x*x+b*x+c


def kappa(x0, x_delta, f):
    top = (np.abs(f(x0+x_delta)-f(x0))*np.abs(x0))
    bot = np.abs(f(x0*x_delta))
    return top/bot


x_0, x_delta = -1, 10e-5
a, b, c = np.float64(10e-5), np.float64(10e3), np.float64(10e3)
_n = 6

widths = [10e1, 10e0, 10e-1, 10e-2, 10e-3, 10e-4]
ints = [np.linspace(x_0-widths[i], x_0+widths[i], 1000) for i in range(_n)]
kappas = [kappa(ints[i], x_delta, poly) for i in range(_n)]
real_kappa = 2*a*x_0 + b

for i in range(_n):
    min1, max1 = real_kappa, real_kappa
    min2, max2 = np.min(kappas[i]), np.max(kappas[i])
    _min, _max = np.min([min1, min2])*1.1, np.max([max1, max2])*1.1

    plt.plot([ints[i][0], ints[i][1]],
             [real_kappa, real_kappa],
             c='k', label='real kappa')
    plt.plot(ints[i], kappas[i], c='b', label='numerical kappa')
    plt.title(str(_min) + " " + str(_max))
    plt.ylim((_min, _max))
    plt.legend()
    plt.show()


#%% C & D
# nope
