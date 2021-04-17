import numpy as np
import numdifftools as nd
from scipy.optimize import fminbound
from parametros import generar_datos, func_objetivo, gradient_f
from numdifftools import Gradient

ndGrad = Gradient(func_objetivo)
func_lineal = lambda l, x, A, b, dir: func_objetivo(x + l*dir, A, b) 
np.random.seed(220399)

m = 300
n = 50
A, b = generar_datos(m, n)
x0 = np.zeros(n)
#x0 = np.random.randint(-3, 3, size=n)
print("\n")
print(func_objetivo(x0, A, b))
dir = -gradient_f(x0, A, b)
l = fminbound(func_lineal, 0, 1e-4, args=(x0, A, b, dir))
#l = 1e-6
print(l)
print(func_objetivo(x0 + l * dir, A, b))


