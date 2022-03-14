import numpy as np
import scipy
import math
from matplotlib import pyplot as plt
from scipy import integrate

#Units
u0_I = 1
#Integration interval
a= 0.1
b= 0.5

#Total magnetic field
def TMG(z,x):
    A = (u0_I)/(4*math.pi)*(x/(z**2+x**2)**3/2)
    return A

def intervals(a,b,n,L):
    n  = int(2*L/n)
    intervals = np.linspace(a,b,num=n)
    return intervals

#Composite trapezoidal
def simpson3(x):
    y = []
    puntos= np.linspace(0,1, 500)
    for i in puntos:
        y.append(integrate.simpson(TMG(0,i),x))
    return(y)
def simpson4(x):
    y = []
    puntos= np.linspace(0,1, 500)
    for i in puntos:
        y.append(integrate.simpson(TMG(1,i),x))
    return(y)

puntos= np.linspace(0,1, 500)
z0 = simpson3(intervals(a,b,0.01,1))
zL = simpson4(intervals(a,b,0.01,1))

plt.figure(3)
plt.plot(puntos, z0, '--r', color="purple")
plt.plot(puntos, zL, '--r', color="red")
plt.legend(['z=0', 'z=L=1'])
plt.xlabel("X", fontsize = 12)
plt.ylabel("Y", fontsize = 12)
plt.grid()
plt.show()


'''
L = 10
z = L

zL10 =  simpson(a,b,0.01)

plt.figure(4)
plt.plot(intervals(a,b,0.01), z0 , '--r', color="purple")
plt.plot(intervals(a,b,0.01), zL10, '--r', color="red")
plt.legend(['z=0', 'z=L=10'])
plt.grid()
plt.show()
'''