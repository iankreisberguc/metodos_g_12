__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"


# Modulos nativos de python
import numpy as np
import time
import scipy.optimize
import numdifftools as nd

# Modulo creado por nosotros (parametros.py)
from parametros import *

#### Creacion de la funcion objetivo ####

# Constante para a
constant = 250

# Calcula el error entre Ax y b
residuo = lambda x, A, b: np.dot(A, x) - b

# Calcula la funcion de penalizacion
penalizacion = lambda x, a: np.piecewise(x, [abs(x) < a, abs(x) >= a], 
                                    [lambda x: - a**2 * np.log(1 - (x/a)**2),
                                     lambda x: np.inf])

# Calcula la funcion objetivo
func_objetivo = lambda x, A, b, a=constant: np.sum(penalizacion(residuo(x, A, b), a))

def timer(funcion):
    """
    Se crea un decorador (googlear) del tipo timer para testear el tiempo
    de ejecucion del programa
    """
    def inner(*args, **kwargs):

        inicio = time.time()
        resultado = funcion(*args, **kwargs)
        final = round(time.time() - inicio, 3)
        print("\nTiempo de ejecucion total: {}[s]".format(final))

        return resultado
    return inner

def subrutina(x, A, b, Hess, Grad):
    """
    Esta funcion va creando el paso de cada iteracion. Ocupando la teoría
    estudiada. Retorna el valor de la funcion, su gradiente y su hessiano segun
    la iteracion estudiada.
    """
    # Funcion a optimizar, gradiente y hessiano
    valor_FO = func_objetivo(x, A, b)
    gradiente = Grad(x, A, b)
    hessiano = Hess(x, A, b)

    return valor_FO, gradiente, hessiano

def funcion_enunciado(lambda_, x, A, b, direccion_descenso):
    """
    Funcion original evaluada en: x + lambda*direccion_descenso
    """
    # Se actualiza el valor de x
    x = x + lambda_ * direccion_descenso

    return func_objetivo(x, A, b)

@timer
def newton(x0, A, b, epsilon, iteracion_maxima):
    """
    Esta funcion es una aplicacion del metodo de Newton, la que
    va a ir devolviendo valor objetivo, gradiente actual y Hessiano.

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

    # Instancias del gradiente y del hessiano que se utilizan para calcularlos
    Grad = nd.Gradient(func_objetivo)
    Hess = nd.Hessian(func_objetivo)

    # Se prepara el output del codigo para en cada iteracion
    # entregar la informacion correspondiente
    print("\n\n*********       METODO DE NEWTON      **********\n")
    print("ITERACION     VALOR OBJ      NORMA        LAMBDA")

    # Se inicia el ciclo para las iteraciones maximas seteadas por el usuario
    while (stop == False) and (iteracion <= iteracion_maxima):

        # 2º paso del algoritmo: Se obtiene la informacion para determinar
        # el valor de la direccion de descenso
        valor, gradiente, hessiano = subrutina(x, A, b, Hess, Grad)
        direccion_descenso = np.dot(-np.linalg.inv(hessiano), gradiente)

        # 3º paso del algoritmo: Se analiza el criterio de parada
        norma = np.linalg.norm(gradiente, ord=2)
        if norma <= epsilon:
            stop = True
    
        else:
        # 4º paso del algoritmo: Se busca el peso (lambda) optimo
            # Se resuelve el subproblema de lambda
            lambda_ = scipy.optimize.fminbound(funcion_enunciado, 0, 10, args=(x, A, b, direccion_descenso))

        # La rutina de Newton muestra en pantalla para cada iteracion:
        # nº de iteracion, valor de la funcion evaluada en el x de la iteracion,
        # la norma del gradiente y el valor de peso de lambda
        retorno_en_pantalla = [iteracion, valor, norma, lambda_]
        print("%12.6f %12.6f %12.6f %12.6f" % (retorno_en_pantalla[0],
        retorno_en_pantalla[1], retorno_en_pantalla[2], retorno_en_pantalla[3]))


        # 5º paso del algoritmo: Se actualiza el valor de x para la siguiente
        # iteracion del algoritmo
        x = x + lambda_ * direccion_descenso
        iteracion += 1

    print("\nSOLUCION:\n", x)

    return retorno_en_pantalla

if __name__ == '__main__':
    # Testeo de Newton, primero se generan datos para la funcion
    m = 300
    n = 50
    A, b = generar_datos(m, n)
    np.random.seed(220399)

    # Se ocupa el vector de "unos" como punto de inicio
    # (notar el salto que pega) de la iteracion 1 a la 2 el valor objetivo
    # -- Queda a tu eleccion que vector ingresar como solucion para la iteracion 1 --
    x0 = np.random.uniform(-3, 3, size=n)
    print(func_objetivo(x0, A, b))

    # Error asociado 10% este caso
    epsilon = 0.1

    # Maximo de iteraciones (para que no quede un loop infinito)
    iteracion_maxima = 100
    newton(x0, A, b, epsilon, iteracion_maxima)