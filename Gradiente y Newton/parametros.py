__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"

# Modificado por J. Vera el 21-03-2021. Correcciones a la generación


# Modulos nativos de python
import numpy as np

# Constante para a
constant = 100

# Calcula el error entre Ax y b
residuo = lambda x, A, b: np.dot(A, x) - b.T

# Calcula la funcion de penalizacion
penalizacion = lambda x, a: np.piecewise(x, [abs(x) < a, abs(x) >= a], 
                                    [lambda x: - a**2 * np.log(1 - (x/a)**2),
                                     lambda x: np.inf])

# Calcula la funcion objetivo
func_objetivo = lambda x, A, b, a=constant: np.sum(penalizacion(residuo(x, A, b), a))

# Calcula el gradiente de la FO analiticamente
def gradient_f(x, A, b):
    a = constant
    aux_vector = a**2 * 2 * residuo(x, A, b) / (a**2 - residuo(x, A, b)**2)
    return np.dot(aux_vector, A)

def generar_datos(m, n):
    """
    Recibe parametros m y n, con m > n, retorna una matriz A de mxn con valores
    entre -10 y 10 y un vector b de mx1 con valores entre -10 y 10
    """
    assert (m > n), "m debe ser mayor a n"
    A = np.random.uniform(-10, 10, size=(m, n))
    b = np.random.uniform(-10, 10, size=m)
    return A, b


if __name__ == '__main__':
    A, c = generar_datos(3, 5)
    print(A)
    print(c)

