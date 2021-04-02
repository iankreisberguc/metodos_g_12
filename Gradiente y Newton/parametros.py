__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"

# Modificado por J. Vera el 21-03-2021. Correcciones a la generación


# Modulos nativos de python
import numpy as np
import scipy.linalg
import random

def generar_datos(dimension):
    """
    Esta funcion crea una matriz cuadrada semidefinida positiva para
    ocuparla en los programas presentados.
    """
    '''
    # Se crea un vector aleatorio positvo segun la dimension entregada
    vector = np.random.random(dimension)

    # Se crea una matriz que contenga en su diagonal
    # el vector ingresado como argumento
    D = np.diag(vector)
    B = np.random.rand(dimension,dimension)
    Q = np.matmul(np.transpose(B),np.matmul(D,B))
    # Por construcción, la matriz Q es invertible 
    b = np.random.rand(dimension, 1)

    # Retorna la matriz y vector listo para ejecutar
    return Q, b
    '''
    lista = []
    for i in range(dimension):
        lista.append(random.uniform(-10,10))

    c = np.array(lista)

    matriz = []
    for i in range(dimension):
        vectores = []
        for j in range(dimension):
            vectores.append(random.uniform(-10,10))
        matriz.append(vectores)

    A = np.array(matriz)
    return A, c
if __name__ == '__main__':
    A, c =generar_datos(5)
    print(A)
    print(c)

