"""
Created on Sat Sep  5 16:57:40 2020
===============
SPSA Algorithm
===============
Algorithm taken from Bhatnagar 2013, Algorithm 5.1
@author: Abhinav
"""
#%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loss(v):
    x, y = v[0], v[1]
    # loss function is x exp(−(x^2+y^2))+(x^2+y^2)/20.
    tmp = x * np.exp(-(x**2+y**2)) + (x**2 + y**2)/20 + np.random.rand()*0.01
    return tmp


class spsa:
    def __init__(self, varnum):
        np.random.seed(123)
        self.theta = np.zeros((varnum,))

        self.iter_limit = 100  # Q or iteration limit
        self.N= varnum
        #step sizes
        self.a, self.d, self.A = .3, .2, 3
        self.alpha, self.gamma = .27, .2
        self.loss_val = []

    def compute(self, loss):
        for i in range(self.iter_limit):
            delta = 2*np.random.binomial(n=1, p=.5, size=self.N) - 1
            ai = (self.a/ (i+1+self.A))**self.alpha
            di = (self.d/(i+1))**self.gamma
            theta_plus = self.theta+di*delta
            theta_minus = self.theta-di*delta
            yplus = loss(theta_plus)
            yminus = loss(theta_minus)
            g_hat = (yplus-yminus) / (2*di*delta)
            self.loss_val.append(loss(self.theta))
            self.theta -= ai*g_hat

        self.plot_results()
        return self.theta

    def plot_results(self):
        plt.plot(self.loss_val)
        x = np.arange(-2,2,.1)
        y = np.arange(-2,2,.1)
        B, D = np.meshgrid(x,y)
        nu=B * np.exp(-(B**2+D**2)) + (B**2 + D**2)/20 + np.random.rand()*0.01
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(B, D, nu)
        ax.plot([self.theta[0]], [self.theta[1]],self.loss_val[-1],  markerfacecolor='r', markeredgecolor='r', marker='o', markersize=17)

        plt.xlabel('b')
        plt.ylabel('d')
        plt.show()

if __name__ == "__main__":
    algo = spsa(2)
    theta=algo.compute(loss=loss)