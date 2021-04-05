__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"


# Modulos nativos de python
import numpy as np
import time
import scipy.optimize

# Modulo creado por nosotros (parametros.py)
from parametros import *

#### Creacion de la funcion objetivo ####

# Constante para a
constant = 1000

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

def subrutina(x, A, b):
    """
    Esta uncion va creando el paso de cada iteracion. Ocupando la teoría
    estudiada. Retorna el valor de la funcion, su gradiente y su hessiano segun
    la iteracion estudiada.
    """
    funcion_objetivo = func_objetivo(x, A, b)

    return funcion_objetivo 



if __name__ == '__main__':
    # Testeo de BFGS, primero se generan datos para la funcion
    m = 300
    n = 50
    
#    np.random.seed(1610)
    np.random.seed(220399)
    
    A, b = generar_datos(m, n)

    # Se ocupa el vector de "unos" como punto de inicio
    # (notar el salto que pega) de la iteracion 1 a la 2 el valor objetivo
    # -- Queda a tu eleccion que vector ingresar como solucion para la iteracion 1 --
    x0 = np.ones(n)

    # Error asociado 10% este caso
    #epsilon = 0.1

    # Maximo de iteraciones (para que no quede un loop infinito)
    #iteracion_maxima = 5

    res = scipy.optimize.minimize(subrutina, x0, (A, b), method='BFGS', options={'gtol': 1e-6, 'disp': True})
    
    print(res.message)
