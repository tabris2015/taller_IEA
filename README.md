# taller_IEA
Repo de código para los ejemplos y ejercicios del taller de Vision y machine learning

## Preparación del entorno

Para poder correr correctamente todos los programas de este repositorio será necesario tener instalados
los siguientes paquetes y librerías:

  - **pip** para poder instalar módulos de python de manera fácil.
  - **numpy** es una librería de cálculo científico optimizada para python, incluye un entorno similar al de MATLAB.
  - **scipy** es un stack con una multitud de funciones útiles para científicos e ingenieros como algoritmos de optmimización, estadística, etc.
  - **matplotlib** es una librería para graficar en python, similar a los comandos plot de MATLAB pero con muchas más opciones.
  - **scikit learn** es la librería que contiene todos los algoritmos de aprendizaje supervisado y no supervisado que necesitamos.
  - **scikit image** una librería con una variedad de algoritmos de procesamiento de imágenes y extracción de característica, contiene funciones de más fácil utilización que las de OpenCV.
  - **OpenCV** una librería con multitud de funciones y algoritmos optimizados en harware para desarrollo de visión artificial, nos facilitará la adquisición de imágenes de una webcam y una variedad de algoritmos de preprocesamiento.
  - **iPython** (opcional) es un intérprete inteactivo de python con varias utilidades como autocompletado de funciones y comandos de sistema.

### Preparando python

Antes de comenzar cualquier trabajo necesitamos configurar el entorno de python. Ingrese a una terminal y escriba los siguientes comandos:

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential python-dev python-devel python-pip
```