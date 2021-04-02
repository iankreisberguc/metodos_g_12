__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"

# Modificado por J. Vera 21-03-2021


# Modulos creados por usuario
from newton import *
from steepest import *
from parametros import *
from numpy import *


# Primero se generan datos para la funcion
# esta informacion es la que pueden cambiar a su antojo
dimension_matriz_Q = 10
iteracion_maxima_newton = 100
iteracion_maxima_gradiente = 100
epsilon = 0.0001


# (si deseas probar un metodo y no ambos comenta la linea 31 o 34, respectivamente)


# Para que siempre genere los mismos datos al azar
np.random.seed(1610)
#np.random.seed(202)

# En base a su informacion entregada como input de aqui en adelante el programa
# se corre solo
Q, c = generar_datos(dimension_matriz_Q)

# Se ocupa el vector de "unos" como punto de inicio
# (notar el salto que pega) de la iteracion 1 a la 2 el valor objetivo
# -- Queda a tu eleccion que vector ingresar como solucion para la iteracion 1 --
x0 = np.random.rand(dimension_matriz_Q, 1)

# Maximo de iteraciones para newton (y asi no quede un loop infinito)

#a es parametro de la funcion con a>0
a = 1
newton(Q, c, x0, epsilon, iteracion_maxima_newton, a)

# Maximo de iteraciones para gradiente (y asi no quede un loop infinito)
#gradiente(Q, c, x0, epsilon, iteracion_maxima_gradiente)

