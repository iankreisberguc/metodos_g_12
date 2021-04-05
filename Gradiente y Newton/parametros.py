__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"

# Modificado por J. Vera el 21-03-2021. Correcciones a la generación


# Modulos nativos de python
import numpy as np

def generar_datos(m, n):
    """
    Recibe parametros m y n, con m > n, retorna una matriz A de mxn con valores
    entre -10 y 10 y un vector b de mx1 con valores entre -10 y 10
    """
    assert (m > n), "m debe ser mayor a n"
    A = np.random.uniform(-10, 10, size=(m, n))
    b = np.random.uniform(-10, 10, size=(m, 1))
    return A, b


if __name__ == '__main__':
    A, c = generar_datos(3, 5)
    print(A)
    print(c)

