__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"


# Modulos nativos de python
import numpy as np
import time
import scipy.optimize

# Modulo creado por nosotros (parametros.py)
from parametros import generar_datos

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


def BFGS(x0, A, b, tol):
    t0 = time.time()
    res = scipy.optimize.minimize(func_objetivo, x0, (A, b), method='BFGS', options={'gtol': tol, 'disp': True})
    t1 = time.time() - t0
    print("\nTiempo de demora de BFGS:", t1, "\n\n")
    print(res.message)
    print()
    #x_opt = res.x
    #print("Solucion BFGS:", x_opt, sep="\n")


if __name__ == '__main__':
    print("\n\n ---------------- BFGS ----------------\n")
    # Testeo de BFGS, primero se generan datos para la funcion
    m = 300
    n = 50
    
#    np.random.seed(1610)
    np.random.seed(220399)
    
    A, b = generar_datos(m, n)

    # Se ocupa el vector de "unos" como punto de inicio
    # (notar el salto que pega) de la iteracion 1 a la 2 el valor objetivo
    # -- Queda a tu eleccion que vector ingresar como solucion para la iteracion 1 --
    x0 = np.zeros(n)
    print("Valor de la FO:", func_objetivo(x0, A, b), "\n")

    # Error asociado 10% este caso
    #epsilon = 0.1

    # Maximo de iteraciones (para que no quede un loop infinito)
    #iteracion_maxima = 5
    t0 = time.time()
    res = scipy.optimize.minimize(func_objetivo, x0, (A, b), method='BFGS', options={'gtol': 1e-6, 'disp': True})
    t1 = time.time() - t0
    print("\nTiempo de demora:", t1, "\n\n")
    print(res.message)
    print()
    x_opt = res.x
    print(np.linalg.norm(residuo(x_opt, A, b)))
    x_opt2 = [-0.00090309,  0.00118256, -0.00100717, -0.00056781, -0.00205902,  0.00368465,
 -0.0022257,   0.00025377, -0.00182315,  0.00117174,  0.00123833,  0.00172285,
  0.00424657, -0.00023223,  0.0037284,   0.00054259, -0.00204912,  0.00091248,
  0.00335923, -0.00336994,  0.00051575, -0.00194586,  0.00196556, -0.00144571,
  0.00215869,  0.00218757, -0.00121628, -0.00055436, -0.00313988,  0.00229273,
 -0.00149506,  0.00287618, -0.00169493, -0.00168946, -0.00079053,  0.00655203,
  0.0007888,  -0.0018132,  -0.00192264,  0.0014942,   0.00222382,  0.00028745,
  0.00208139, -0.00229559, -0.00244832, -0.00284838,  0.00101864, -0.00155116,
 -0.00158425, -0.00217585]
    print(res.x)
