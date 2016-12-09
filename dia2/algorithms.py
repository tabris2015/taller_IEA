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


# RECOMMENDER SYSTEMS ALGORITHMS
# the following functions are used for a collaborative filtering
# algorithm for a recommmender system.
def COFICostFunction(params, Y, R, num_users, num_products, num_features, lambd):

	# params es un vector de dimensiones num_features*num_products*2
	X = params[:num_products*num_features].reshape(num_products, num_features)
	Theta = params[(-num_users*num_features):].reshape(num_users, num_features)
	#COFICOSTFUNC Collaborative filtering cost function
	#   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
	#   num_features, lambda) returns the cost and gradient for the
	#  collaborative filtering problem.
	#
	    
	# You need to return the following values correctly

	J = (1/2.0) * np.sum(np.sum(((np.dot(X,Theta.T)-Y)**2)*R)) +  \
		(lambd/2.0)*np.sum(np.sum(Theta**2)) + \
		(lambd/2.0)*np.sum(np.sum(X**2))

	# =============================================================

	return J

def COFIGradients(params, Y, R, num_users, num_products, num_features, lambd):
	# params es un vector de dimensiones num_features*num_products*2
	X = params[:num_products*num_features].reshape(num_products, num_features)
	Theta = params[(-num_users*num_features):].reshape(num_users, num_features)

	X_grad = np.zeros(X.shape);
	Theta_grad = np.zeros(Theta.shape);

	X_grad = np.dot(((np.dot(X,Theta.T) - Y) * R), Theta) + lambd * X
	Theta_grad = np.dot(((np.dot(X,Theta.T) - Y) * R).T, X) + lambd * Theta
	
	params_grad = np.concatenate((X_grad.flatten(), Theta_grad.flatten()))
	return params_grad


def COFINormalizeRatings(Y, R):
	(m,n) = Y.shape
	Ymean = np.zeros((m,1))
	Ynorm = np.zeros(Y.shape)
	for i in range(m):
		idx = R[i,].nonzero()
		Ymean[i] = np.mean(Y[i,].take(idx))
		for j in idx:
			Ynorm[i,j] = Y[i,j] - Ymean[i]
	return Ynorm, Ymean


def COFIGradientDescent(params, Y, R, num_users, num_products, num_features, lambd, alpha, num_iters):
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

	J_history = np.zeros(num_iters)	#vector con los costos
				#vector con las derivadas
	for it in range(num_iters):
		
		params = params - alpha*COFIGradients(params, Y, R, num_users, num_products, num_features, lambd)
		#print "theta:",theta
		J_history[it] = COFICostFunction(params, Y, R, num_users, num_products, num_features, lambd)
		print it, " costo:" ,J_history[it]
		#J_history[iter] = computeCost
	
	return (params, J_history)

