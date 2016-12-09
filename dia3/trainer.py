#This code is beerware; if you use it, please buy me 
#a cold beverage next time you run into one of
#us at the local.
#julio de 2015- Jose Laruta - Instituto de Electronica Aplicada
#Code developed in python 2.7.3
#****************************************************************/

#modulos necesarios
import os

from sklearn.externals import joblib	#utilidad para salvar datos
from sklearn import datasets			# "  "    para importar el dataset
from skimage.feature import hog 		# funcion para extraer hog
from sklearn.svm import LinearSVC		# clasificador SVM
from sklearn.naive_bayes import GaussianNB	#clasificador Naive-Bayes
import numpy as np 						
from matplotlib import pyplot as plt
import time
import cv2
import random

import affinity, multiprocessing 	# para todos los nucleos (np)

affinity.set_process_affinity_mask(0,2**multiprocessing.cpu_count()-1)


print "importando dataset..."


#descargamos dataset para digitos escritos a mano
dataset = datasets.fetch_mldata("MNIST Original")


#separamos los features del dataset y las etiquetas en arrays distintos
features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')

examples = []

# extraer 12 imagenes aleatorias del dataset
for i in range(100):
	examples.append(random.choice(features).reshape((28,28)))

examples = np.array(examples)
print examples.shape

#imagen de prueba 10x10
ex_matrix = np.zeros((28,308))

for i in range(10):
	aux_row = np.zeros((28,28))
	for j in range(10):
		aux_row = np.concatenate((aux_row,examples[j+10*i]), axis=1)
	ex_matrix = np.concatenate((ex_matrix, aux_row), axis=0)

print "tamano matriz de prueba: " + str(ex_matrix.shape)


while(1):
	cv2.imshow("ejemplos", ex_matrix)
	tecla = cv2.waitKey(5) & 0xFF
	if tecla == 27:
		break

cv2.destroyAllWindows()

print "dataset importado!"

#extraemos los features HOG y almagenamos en otro array
print "extrayendo caracteristicas HOG..."
start_time = time.time()
	
list_hog_fd = []

for feature in features:
	fd = hog(
			feature.reshape((28, 28)), 		
			orientations=9, 				#direcciones
			pixels_per_cell=(14, 14), 		# 4 bloques
			cells_per_block=(1, 1), 
			visualise=False
		)
	list_hog_fd.append(fd)
	
hog_features = np.array(list_hog_fd, 'float64')

print "terminada extraccion de descriptores en " + str(time.time() - start_time) + " seg."
	
print "terminado!" + str(hog_features.shape)

opcion = int(input("seleccione algoritmo de aprendizaje"))

if opcion == 1:
	#creamos el clasificador
	print "entrenando SVM..."
	start_time = time.time()
	
	clf = LinearSVC()

	#entrenamiento
	clf.fit(hog_features, labels)
	
	print "terminado entrenamiento en " + str(time.time() - start_time) + " seg."
	
elif opcion == 2:
	#creamos el clasificador
	print "entrenando Naive Bayes..."
	start_time = time.time()
	
	clf = GaussianNB()

	#entrenamiento
	clf.fit(hog_features, labels)
	
	print "terminado entrenamiento en " + str(time.time() - start_time) + " seg."

else:
	print "opcion no valida"

#guardamos el clasificador en un archivo
joblib.dump(clf, "digits_cls.pkl", compress=3)

