#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:05:02 2020

@author: jorgereyesh
"""
import numpy as np

def ode_RK4(f, X_0, dt, T):
    """ 
    Solve the ode u' = f(u(t),t) for t in [0, T] 
    (u'1, u'2,...,) = (f1, f2,...)
    
    """
    import numpy as np
    
    N_t = int(round(T/dt))
    #  Create an array for the functions ui 
    u = np.zeros((N_t+1,len(X_0))) 
    t = np.linspace(0, N_t*dt, len(u))
    # Initial conditions
    for j in range(len(X_0)):
        u[j,0] = X_0[j]
        
    # RK4
    for n in range(N_t):
        u1 = f(u[n] + 0.5*dt* f(u[n], t[n]), t[n] + 0.5*dt)
        u2 = f(u[n] + 0.5*dt*u1, t[n] + 0.5*dt)
        u3 = f(u[n] + dt*u2, t[n] + dt)
        u[n+1] = u[n] + (1/6)*dt*( f(u[n], t[n]) + 2*u1 + 2*u2 + u3)
    
    return u, t
   
    
def demo_exp():
    """ Resolve u'=u """
    import matplotlib.pyplot as plt
    
    def f(u,t):
        return np.asarray([u])

    u, t = ode_RK4(f, [1] , 0.1, 1.5)
    
    plt.plot(t, u[0,:],"b*", t, np.exp(t), "r-")
    plt.show()
    
def demo_osci():
    """ 
    Resolve x'' + omega**2 x = 0.
    Let u1 = x; u2 = x' = u1'.
    Then the system is:
        u1' = u2
        u2' = -omega**2 u1
        
    Then if u = (u1, u2)
    We need to solve u' = f(u,t)
    where f(u,t) = (u2, -omega**2 u1)
    """
    import matplotlib.pyplot as plt
    
    omega = 2 
    def f(u,t):
        omega =2 
        u, v = u
        return np.asarray([v, - omega**2*u])
    
    X_0 = [2,0]
    u, t = ode_RK4(f, X_0, 0.1, 2)
    U = u[:,0]
    V = u[:,1]
    """
    S = [U,V]
    for i in [0]:
        plt.plot(t, S[i], "b--", t, 2*np.cos(omega*t), "r")
    plt.show()
    """
    
    return U, V, omega

def solvediff():
    import matplotlib.pyplot as plt
    
    def f(u,t):
        return t**3 + 2*t + t**2*((1 + 3*t**2)/1 + t + t**3) - t*u - ((1+3*t**2)/(1+t+t**3))*u
    u,t = ode_RK4(f, [1], 0.1, 2)
    
    def analitic(x):
        return ((np.exp((-x**2)/2.))/(1. + x + x**3)) + x**2
    
    
    plt.figure(figsize=(20,10))
    plt.plot(t,u,"r*")
    plt.plot(t, analitic(t))
    plt.show()
    