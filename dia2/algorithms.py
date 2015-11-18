# funciones basicas para algoritmos de aprendizaje supervisado
import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as op

def computeCost(X, y, theta):
	"""
		funcion para calcular el costo para la regresion lineal
		esta funcion acepta 3 parametros:
		- X es un arreglo numpy de dimensiones (m,n)
		  que contiene los ejemplos de entrenamiento
		- y es un arreglo numpy unidimensional que 
		  contiene las salidas
		- theta es un arreglo numpy unidimensional 
		  que contiene los coeficientes de la funcion

		la funcion retorna un escalar que es el costo 

	"""
	m, n = X.shape # numero de datos de entrenamiento

	J = (1/(2.0*m))*np.dot(np.dot(X,theta.T)-y, np.dot(X,theta.T)-y)
	
	return J

#funcion para correr el algoritmo de descenso de gradiente
def gradientDescent(X, y, theta, alpha, num_iters):
	""" 
		funcion para ejecutar el algoritmo de descenso de 
		gradiente, esta funcion encuentra los valores optimos
		para el modelo.
		ENTRADAS:
			- X: vector de entrenamiento
			- y: vector de salidas
			- theta: vector de parametros del modelo
			- alpha: factor de aprendizaje
			- num_iters: numero de iteraciones para el algoritmo
		SALIDAS:
		Devuelve una tupla con 2 arrays:
			- theta: parametros sintonizados
			- J_history: vector con la evolucion de los costos
	"""
	m,n = X.shape			#dimensiones
	J_history = np.zeros(num_iters)	#vector con los costos
	delta = np.zeros(n)					#vector con las derivadas
	for it in range(num_iters):
		for i in range(n):
			delta[i] = (1.0/m)*np.sum((np.dot(X,theta.T).T.flatten() - y)*X[:,i])

		#print "delta:",delta
		theta = theta - alpha*delta
		#print "theta:",theta
		J_history[it] = computeCost(X, y, theta)
		print it, " costo:" ,J_history[it]
		#J_history[iter] = computeCost
	
	return (theta, J_history)

#funcion para normalizar los features
def featureNormalize(X,y):
	m,n = X.shape
	X_norm = X
	mu_x = np.mean(X,axis=0)	
	sigma_x = np.std(X, axis=0)
	mumatx = np.ones((m,n))*mu_x
	sigmamatx = np.ones((m,n))/sigma_x
	X_norm = sigmamatx*(X-mumatx)

	mu_y = np.mean(y,axis=0)	
	sigma_y = np.std(y, axis=0)
	mumaty = np.ones(m)*mu_y
	sigmamaty = np.ones(m)/sigma_y
	y_norm = (sigmamaty*(y-mumaty))

	return (X_norm,mu_x,sigma_x,y_norm,mu_y,sigma_y)

#funcion con la ecuacion normal para la regresion lineal

def predict(x,theta, params):
	x_norm = x/params[1] - params[0]
	x_norm = np.concatenate((np.ones(1),x_norm), axis=1)
	h = np.dot(x_norm,theta)
	return np.array([h,(h + params[3])*params[2]])

def normalEqn(X,y):
	theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),y)
	return theta

def plotDataLog(x,y):
	pass

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def costFunction(theta, X, y):
	m,n = X.shape
	h = sigmoid(np.dot(X,theta))
	J = -(1.0/m) * np.sum(y*np.log(h)+(1-y)*np.log(1-h))
	return J

def gradient(theta, X, y):
	m,n = X.shape
	h = sigmoid(np.dot(X,theta))
	grad = (1.0/m) * np.dot(h - y,X)
	return grad

def logPredict(theta, X):
	m, n = X.shape
	return sigmoid(np.dot(X,theta)) >= 0.55

def costFunctionReg(theta, X, y, lamb):
	m,n = X.shape
	h = sigmoid(np.dot(X,theta))
	reg = (lamb/(2.0*m)) * np.dot(theta[1:],theta[1:])
	J = -(1.0/m) * np.sum(y*np.log(h)+(1-y)*np.log(1-h)) + reg
	return J

def gradientReg(theta, X, y, lamb):
	m,n = X.shape
	h = sigmoid(np.dot(X,theta))
	reg = (lamb/m) * theta[1:]
	grad = (1.0/m) * np.dot(h - y,X)
	grad[1:] = grad[1:] + reg

	return grad

