import numpy as np
import scipy
import math
from matplotlib import pyplot as plt
from scipy import integrate

p = 3
u_i = 1
L = 1
z = 0
#Integration interval
a= 1
b= 5

#Total magnetic field
def TMG(z,x):
    constante = u_i/(4*math.pi)
    integral = x/((z**2+x**2)**3/2)
    return constante*integral

#Composite trapezoidal
def simpson(a,b,n):
    h  = int((b-a)/n)
    intervals = np.linspace(a,b,num=h)
    intervals1 = []
    for x_l in intervals: intervals1.append(x_l/L)
    Area_vector=[]
    for i in range (0,len(intervals1)-1,1):
        subintervals = np.arange(intervals1[i], intervals1[i+1], (intervals1[i+1]-intervals1[i])/p)
        y=[]
        for j in subintervals:
            y.append(TMG(z,j))#z=0
        a = integrate.simpson(y,x=subintervals)
        Area_vector.append(a)
    Area_vector.append(a)
    print(subintervals)
    print(y)
    print(Area_vector)
    return(Area_vector)


def intervals(a,b,n):
    h  = int((b-a)/n)
    intervals = np.linspace(a,b,num=h)
    return intervals

z0 = simpson(a,b,0.01)
"""
z = L
zL = simpson(a,b,0.01)

plt.figure(3)
plt.plot(intervals(a,b,0.01), z0, '--r', color="purple")
plt.plot(intervals(a,b,0.01), zL, '--r', color="red")
plt.legend(['z=0', 'z=L=1'])
plt.xlabel("X", fontsize = 12)
plt.ylabel("Y", fontsize = 12)
plt.grid()
plt.show()
"""
L = 10
z = L

zL10 =  simpson(a,b,0.01)

plt.figure(4)
plt.plot(intervals(a,b,0.01), z0 , '--r', color="purple")
plt.plot(intervals(a,b,0.01), zL10, '--r', color="red")
plt.legend(['z=0', 'z=L=10'])
plt.grid()
plt.show()