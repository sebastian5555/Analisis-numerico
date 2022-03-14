import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constantes

m = 100000
k = 2e-6
y0 = 1000
a = 0
b = 30
n = 60
#()
# Funcion

def f_yt(y, t , k, m):
    return k*(m-y)*y

# Analitica

def y(t, m):
    return m / (1 + 99* np.exp(-0.2 * t))

# Metodos numericos

def Euler(f, a, b, y0, n, k, m):
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0] = a
    y[0] = y0
    h = (b - a) / n
    for i in range(n):
        t[i + 1] = t[i] + h
        y[i + 1] = y[i] + h * f(y[i], t[i], k, m)
    return t, y 

def Euler_Mejorado(f, a, b, y0, n, k, m):
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0] = a
    y[0] = y0
    h = (b - a) / n
    for i in range(n):
        fn = f(y[i], t[i], k, m)
        fhn = f(y[i] + h * fn, t[i] + h, k, m)
        t[i + 1] = t[i] + h
        y[i + 1] = y[i] + (fn + fhn)*(h/2) 
    return t, y 

def Runge_Kutta(f, a, b, y0, n, k, m):
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0] = a
    y[0] = y0
    h = (b - a) / n
    for i in range(n):
        k1 = f(y[i], t[i], k, m)
        k2 = f(y[i] + (h / 2)* k1, t[i] + (h / 2), k, m)
        k3 = f(y[i] + (h / 2)* k2, t[i] + (h / 2), k, m)
        k4 = f(y[i] + h* k3, t[i] + h, k, m)
        t[i + 1] = t[i] + h
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4)* (h/6)
    return t, y 

# Punto a

T_Euler, Y_Euler= Euler(f_yt, a, b, y0, n, k, m)
T_Euler_M, Y_Euler_M= Euler_Mejorado(f_yt, a, b, y0, n, k, m)
T_Runge_K, Y_Runge_K= Runge_Kutta(f_yt, a, b, y0, n, k, m)
y = [y(i, m) for i in T_Runge_K]

'''
# Grafica w(t) vs t
plt.figure(1)
plt.plot(T_Runge_K, y, 'r--' ,color="black")
plt.plot(T_Euler, Y_Euler, 'r--' , color="blue")
plt.plot(T_Euler_M, Y_Euler_M, 'r--' , color="red")
plt.plot(T_Runge_K, Y_Runge_K, 'r--' , color="yellow")
plt.xlabel("X", fontsize = 12)
plt.ylabel("Y", fontsize = 12)
plt.legend(['Analitica', 'Euler', 'Euler Mejorado', 'Runge Kutta'])
plt.grid()
plt.show()


# Grafica |w(t) - y(t)| vs t
plt.figure(2)
plt.plot(T_Euler, np.absolute(Y_Euler - y) , 'b' , color="yellow")
plt.plot(T_Euler_M, np.absolute(Y_Euler_M - y), 'b' , color="red")
plt.plot(T_Runge_K, np.absolute(Y_Runge_K - y), 'b' , color="black")
plt.xlabel("X", fontsize = 12)
plt.ylabel("Y", fontsize = 12)
plt.legend(['Euler', 'Euler Mejorado', 'Runge Kutta'])
plt.grid()
plt.show()

'''
# Grafica |w(t) - y(t)|/y(t) vs t
plt.figure(3)
plt.plot(T_Euler, np.divide(np.absolute(Y_Euler - y), np.absolute(y)) , 'b' , color="yellow")
plt.plot(T_Euler_M, np.divide(np.absolute(Y_Euler_M - y), np.absolute(y)), 'b' , color="red")
plt.plot(T_Runge_K, np.divide(np.absolute(Y_Runge_K - y), np.absolute(y)), 'b' , color="black")
plt.xlabel("X", fontsize = 12)
plt.ylabel("Y", fontsize = 12)
plt.legend(['Euler', 'Euler Mejorado', 'Runge Kutta'])
plt.grid()
plt.show()


# Grafica |w(t) - y(t)| vs h
new_h = np.linspace(0.0005, 0.5, 100)


plt.figure(2)
plt.plot(T_Euler, np.absolute(Y_Euler - y) , 'b' , color="yellow")
plt.plot(T_Euler_M, np.absolute(Y_Euler_M - y), 'b' , color="red")
plt.plot(T_Runge_K, np.absolute(Y_Runge_K - y), 'b' , color="black")
plt.xlabel("X", fontsize = 12)
plt.ylabel("Y", fontsize = 12)
plt.legend(['Euler', 'Euler Mejorado', 'Runge Kutta'])
plt.grid()
plt.show()

