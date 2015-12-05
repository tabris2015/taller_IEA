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

**Instalamos pip**
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential python-dev python-devel python-pip gcc gfortran libblas-dev liblapack-dev cython
```
**Instalamos numpy**

```bash
sudo pip install --upgrade numpy
```

**Instalamos scipy**

```bash
sudo pip install --upgrade scipy
```

**Instalamos matplotlib**
```bash
sudo pip install --upgrade matplotlib
```

**Probando la instalación**
Para verificar si hemos instalado todo correctamente escribimos en una terminal:
```bash
python
```
se abrirá el intérprete de python:

```
Python 2.7.6 (default, Jun 22 2015, 17:58:13) 
[GCC 4.8.2] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>
```

una vez dentro intentamos importar los paquetes instalados:

```python
import numpy
import scipy
import matplotlib.pyplot as plt
```
si logramos importar sin ningún problema quiere decir que la instalación de numpy ha sido exitosa. Ahora podemos ejecutar los ejemplos del día 2 concercientes a regresión lineal y regresión logística.

**Instalando Scikit Learn y Scikit Image**

La instalación de estos paquetes es exactamente igual a los anteriores:

```bash
sudo pip install --upgrade scikit-learn
sudo pip install --upgrade scikit-image
```

De la misma manera, para probar el funcionamiento usamos `import sklearn, skimage` y verificamos que se importen de manera correcta.

### (opcional) Instalando iPython y iPython notebook
IPython es una herramienta muy útil para quienes quieren hacer su desarrollo en python más versátil, posee muchas funciones, pueden ver mas en su [página oficial](www.ipython.org). Por su parte, iPython notebook es otra herramienta que nos permite iniciar sesiones interactivas de python en un navegador web y poder almacenarlas para después compartirlas, es muy útil cuando se trata de compartir código o hacer tutoriales. Para instalar ambas herramientas abrimos una terminal y escribimos los siguientes comandos:

```bash
sudo pip install ipython
sudo pip install ipython[notebook]
```

### Compilando OpenCV
Una de las librerías más útiles y ampliamente utilizadas en el desarrollo de aplicaciones de visión artificial es OpenCV, es una librería completamente libre y de código abierto, mantenida por miles de desarrolladores y lo más interesante es que está optimizado a nivel de hardware, con funciones implementadas de la mejor manera y soporta distintas plataformas y lenguajes de programación como c, c++, java y por su puesto python.

Si bien esta herramienta es indispensable para poder hacer aplicaciones de visión artificial sin perder tiempo en desarrollar nuestras propias funciones, es necesario instalarlo de manera correcta. En los repositorios de ubuntu existe una versión precompilada, pero no es recomendable usarla pues es una versión muy antigua de opencv. En lugar de instalar el paquete de los repos oficiales procederemos a compilar la librería desde el código fuente.

> para instalar opencv en python necesitamos tener instalado antes numpy

**Instalando dependencias**
Antes de instalar OpenCV en sí, necesitamos que nuestro sistema cumpla con algunas dependencias, éstos paquetes son librerías y controladores necesarios para que OpenCV funcione correctamente. Abrimos una terminal y escribimos:

```bash
sudo apt-get install build-essential libgtk2.0-dev libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen3-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev default-jdk ant libvtk5-qt4-dev
```
**Compilando OpenCV**
Una vez tengamos las dependencias instaladas procederemos a descargar el código fuente de OpenCV desde su repositorio oficial en Sourceforge. La versión que utilizaremos en este tutorial será la 2.4.11 por ser la más estable y utilizada. Sería un buen ejercicio migrar los ejemplos a la version 3.0

Abrimos una terminal, nos situamos en la carpeta del usuario y descargamos el código fuente:

```bash
cd ~
wget http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.11/opencv-2.4.11.zip
unzip opencv-2.4.11.zip
cd opencv-2.4.11.zip
```

Una vez descargado el código fuente necesitaremos generar un Makefile usando cmake. En este archivo definiremos las partes de OpenCV que queremos compilar. para esto creamos un directorio llamado *build* y dentro generamos el Makefile:

```bash
mkdir build
cd build
cmake -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_VTK=ON ..
```
luego de ejecutar debemos revisar el texto que aparece en la terminal verificando que no exista ningún error. Si es el caso, entonces ya estamos listos para compilar OpenCV 2.4.11:


```bash
make
```

este último comando, dependiendo de nuestro hardware tomará un buen tiempo, en mi portátil toma alrededor de una hora para compilar.
Luego de eso escribimos:

```bash
sudo make install
```

Ahora debemos configurar la librería escribiendo en un fichero llamado opencv.conf:
```
sudo gedit /etc/ld.so.conf.d/opencv.conf
```

cuando abramos el archivo, escribimos la siguiente línea y guardamos:

```
/usr/local/lib
```

y luego ejecutamos en la terminal:

```bash
sudo ldconfig
```

Ahora debemos editar un archivo más:

```
sudo gedit /ect/bash.bashrc
```

scrollear hasta el final del archivo y añadir las siguientes 2 líneas abajo de todo:

```bash
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH
```

Y listo! ya tenemos OpenCV instalado y configurado. Para probarlo necesitamos cerrar la terminal y reiniciar el equipo.
