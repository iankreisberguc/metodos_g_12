__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"


# Modulos nativos de python
import numpy as np
import time
import scipy.optimize

# Modulo creado por usuario
from parametros import *

# Se crea un decorador (googlear) del tipo timer para testear el tiempo
# de ejecucion del programa
def timer(funcion):
    def inner(*args, **kwargs):

        inicio = time.time()
        resultado = funcion(*args, **kwargs)
        final = round(time.time() - inicio, 3)
        print("\nTiempo de ejecucion total: {}[s]".format(final))

        return resultado
    return inner

# Se define la evaluacion de valores dentro de cada iteracion
# de la rutina del gradiente
def subrutina(Q, c, x):
    """
    Esta funcion va creando el paso de cada iteracion. Ocupando la teoría
    estudiada. Retorna el valor de la funcion, su gradiente segun
    la iteracion estudiada.
    """
    # Dimensiones de la matriz ingresada
    n, _ = Q.shape

    # Se crean las variables que ayudan a definir la funcion principal
    # para este caso el vector que solo posee valor en coordenada n
    # y la matriz que posee valor en coordenada en fila m y coordenada n
    # ademas del valor del real alpha

    vector_canonico = np.zeros((n, 1))
    vector_canonico[n-1][0] = 1

    matriz_canonica = np.zeros((n,n))
    matriz_canonica[n-1][n-1] = 1

    alpha = 10

    # Funcion a optimizar, gradiente y hessiano
    funcion_objetivo = 0.5 * np.dot(np.transpose(x), np.dot(Q,x)) + np.dot(np.transpose(c), x) + alpha*(5 - x[n-1])**4
    gradiente = np.dot(Q, x) + c + vector_canonico * (-4*alpha*(5-x[n-1])**3)

    return funcion_objetivo, gradiente

def funcion_enunciado(lambda_, Q, c, x, alpha, direccion_descenso):
    """
    Funcion original evaluada en: x + lambda*direccion_descenso
    """
    m, n = Q.shape

    # Se actualiza el valor de x
    x = x + lambda_*direccion_descenso

    return (0.5 * np.dot(np.transpose(x), np.dot(Q,x)) + np.dot(np.transpose(c), x) + alpha*(5 - x[n-1])**4)[0][0]

@timer
def gradiente(Q, c, x0, epsilon, iteracion_maxima):
    """
    Esta funcion es una aplicacion del metodo del gradiente, la que
    va a ir devolviendo valor objetivo, gradiente actual.

    Su entrada posee:
    - Q : matriz cuadrada que constituye la funcion definida
    - c : vector asociado que constituye la funcion definida
    - x0 : punto inicial de prueba
    - epsilon : error/ tolerancia deseada
    - iteracion_maxima : numero maximo de iteraciones

    Su retorno (salida) es:
    - valor : valor de la funcion evaluada en x en la iteracion actual
    - x : solucion en la que se alcanza el valor objetivo
    - R : matriz con la informacion de cada iteracion. Es una fila por iteracion
          y esta constituida por:
          - Numero de iteracion
          - valor
          - norma del gradiente
          - paso (lambda)
    """
    # 1º paso del algoritmo: Se definen los parametros iniciales
    iteracion = 0
    stop = False
    x = x0
    alpha = 10

    # Se prepara el output del codigo para en cada iteracion
    # entregar la informacion correspondiente
    print("\n\n*********      METODO DE GRADIENTE      **********\n")
    print("ITERACION     VALOR OBJ      NORMA        LAMBDA")

    # Se inicia el ciclo para las iteraciones maximas seteadas por el usuario
    while (stop == False) and (iteracion <= iteracion_maxima):

        # 2º paso del algoritmo: Se obtiene la informacion para determinar
        # el valor de la direccion de descenso
        [valor, gradiente] = subrutina(Q, c, x)
        direccion_descenso = -gradiente

        # 3º paso del algoritmo: Se analiza el criterio de parada
        norma = np.linalg.norm(gradiente, 2)

        if norma <= epsilon:
            stop = True
        else:
        # 4º paso del algoritmo: Se busca el peso (lambda) optimo
            # Se definen las dimensiones
            [m, n] = Q.shape

            # Se resuelve el subproblema de lambda
            lambda_ = scipy.optimize.fminbound(funcion_enunciado, 0, 10, args=(Q, c, x, alpha, direccion_descenso))

        # La rutina del gradiente muestra en pantalla para cada iteracion:
        # nº de iteracion, valor de la funcion evaluada en el x de la iteracion,
        # la norma del gradiente y el valor de peso de lambda
        retorno_en_pantalla = [iteracion, valor, norma, lambda_]
#       Nota de J. Vera: Esta forma de "print" requiere Python 3.6 
#        print(f"{retorno_en_pantalla[0]: ^12d}{retorno_en_pantalla[1][0][0]: ^12f} {retorno_en_pantalla[2]: ^12f} {retorno_en_pantalla[3]: ^12f}")

        print("%12.6f %12.6f %12.6f %12.6f" % (retorno_en_pantalla[0],retorno_en_pantalla[1][0][0],retorno_en_pantalla[2],retorno_en_pantalla[3]))


        # 5º paso del algoritmo: Se actualiza el valor de x para la siguiente
        # iteracion del algoritmo
        x = x + lambda_*direccion_descenso
        iteracion += 1

    return retorno_en_pantalla


