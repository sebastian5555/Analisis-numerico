import numpy as np
import scipy
import math
from matplotlib import pyplot as plt
from scipy import integrate

#Trapezoidal constant
p = 3 
#Units
u0_I = 1
L = 1
#Integration interval
a= 1
b= 5*L

#Total magnetic field
def TMG(z,x):
    constante = u0_I/(4*math.pi)
    integral = x/((z**2+x**2)**3/2)
    return constante*integral

#Composite trapezoidal
def simpson(a,b,n):
    h  = int((b-a)/n)
    intervals = np.linspace(a,b,num=h)
    Area_vector=[]
    for i in range (0,len(intervals)-1,1):
        subintervals = np.arange(intervals[i], intervals[i+1], (intervals[i+1]-intervals[i])/p)
        y=[]
        for j in subintervals:
            y.append(TMG(0,j))#Z=0
        a = integrate.simpson(y,x=subintervals)
        Area_vector.append(a)
    Area_vector.append(a)
    return(Area_vector)


def intervals(a,b,n):
    h  = int((b-a)/n)
    intervals = np.linspace(a,b,num=h)
    return intervals

plt.figure(2)
plt.plot(intervals(a,b,0.01), simpson(a,b,0.01),'--r',marker='*', color="yellow")
plt.plot(intervals(a,b,0.05), simpson(a,b,0.05),'--r', marker='*', color="blue")
plt.plot(intervals(a,b,0.1), simpson(a,b,0.1),'--r', marker='*', color="purple")
plt.plot(intervals(a,b,0.5), simpson(a,b,0.5),'--r', marker='*', color="red")
plt.plot(intervals(a,b,1), simpson(a,b,1), '--r',marker='*', color="green")
plt.xlabel("X", fontsize = 12)
plt.ylabel("Y", fontsize = 12)
plt.legend(['∆z=0.01', '∆z=0.05', '∆z=0.1', '∆z=0.5', 'Standart' ])
plt.grid()
plt.show()
