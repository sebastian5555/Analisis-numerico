import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import math

#Units
u0_I = 1
L = 1
#Integration interval
a= 0.1
b= 0.17

#Total magnetic field
def TMG(z,x):
    A = (u0_I)/(4*math.pi)*(x/(z**2+x**2)**3/2)
    return A

def intervals(a,b,dz):
    n  = int(2/dz)
    intervals = np.linspace(a,b,num=n)
    return intervals

#Composite trapezoidal
def simpson(x):
    y=[]
    puntos= np.linspace(0,1, 500)
    for i in puntos:
        y.append(integrate.simpson(TMG(x,i), x))
    return(y)

T1 = simpson(intervals(a,b,0.01))
T2 = simpson(intervals(a,b,0.05))
T3 = simpson(intervals(a,b,0.1))
T4 = simpson(intervals(a,b,0.5))
T5 = simpson(intervals(a,b,1))
puntos= np.linspace(0,1, 500)

#Graphics
plt.figure(2)
plt.plot(puntos, T1,'--r', color="yellow")
plt.plot(puntos, T2,'--r', color="blue")
plt.plot(puntos, T3,'--r', color="purple")
plt.plot(puntos, T4,'--r', color="red")
plt.plot(puntos, T5, '--r', color="green")
plt.xlim([0, 0.30])
plt.xlabel("X", fontsize = 12)
plt.ylabel("Y", fontsize = 12)
plt.legend(['∆z=0.01', '∆z=0.05', '∆z=0.1', '∆z=0.5', 'Standart' ])
plt.grid()
plt.show()
