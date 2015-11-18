# linear_reg.py
# programa para efectuar una regresion lineal con multiples 
# variables usando el algoritmo del descenso de gradiente
# Jose Laruta - noviembre 2015

import numpy as np
from algorithms import *
import matplotlib.pyplot as plt 


# importamos los datos desde el archivo de texto
data = np.loadtxt("ex1data2.txt", delimiter=",",dtype=np.float)

#almacenamos en arreglos numpy
X_orig = data[:,:-1] 
y_orig = data[:,-1:].flatten()
y = y_orig
# normalizamos entrada y salida

X, mu_x, sigma_x, y_1, mu_y, sigma_y = featureNormalize(X_orig,y_orig)

parametros = (mu_x, sigma_x, mu_y, sigma_y)

m,n = X.shape
# aumentamos una columna de unos para vectorizacion
X = np.concatenate((np.ones((m,1)),X),axis=1)

# parametros de aprendizaje

num_it = 30
theta = np.zeros(n+1)

theta_arr = []	
#print computeCost(X,y,theta)
J_hist_arr = []
theta_arr = []
alphas = range(1,10,6)


for alpha in alphas:
	theta, J_hist = gradientDescent(X, y, theta, alpha*0.1, num_it)
	J_hist_arr.append(J_hist)
	theta_arr.append(theta)


## normal eqn
theta_eqn = normalEqn(X,y)
print "gradient descent: ", theta_arr[1]
print "normal eq:",theta_eqn
pr_gd = predict(np.array([2200,3]),theta_arr[1],parametros)
pr_ne = predict(np.array([2200,3]),theta_eqn,parametros)
print "diferencia:",(theta_eqn - theta_arr[1])

"""
plt.subplot(2,1,1)
plt.title("regresion lineal")
#plt.text(80,.40, 'alpha=0.01')
plt.plot(range(num_it),J_hist_arr[0])
plt.ylabel("costo")

plt.subplot(2,1,2)
plt.plot(range(num_it),J_hist_arr[1])
plt.xlabel("iteraciones")
plt.ylabel("costo")


plt.show()
"""
