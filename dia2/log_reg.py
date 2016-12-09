import numpy as np
from algorithms import *
import matplotlib.pyplot as plt 
import scipy.optimize as op

data = np.genfromtxt("ex2data1.txt", delimiter=",",dtype=np.float)

X = data[:,:-1] 
y = data[:,-1:].flatten()

arr1 = np.array(range(100))
arr2 = np.array(range(100))

plt.plot(arr1,arr2)
plt.show()
m,n = X.shape
initial_theta = np.zeros(n) # dim n
lamb = 0.9####

# optimizacion
Result = op.minimize(fun=costFunctionReg,
					x0=initial_theta,
					args=(X, y, lamb),
					method = 'TNC',
					jac = gradientReg)
optimal_theta = Result.x;

print optimal_theta
print logPredict(optimal_theta,np.array([[20,30],[45, 85]]))
