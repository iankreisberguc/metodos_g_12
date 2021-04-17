from newton import newton
from steepest import gradiente
from parametros import generar_datos, func_objetivo
import numpy as np
from Test_bfgs import BFGS

m = 300
n = 50
np.random.seed(220399)
A, b = generar_datos(m, n)
x0 = np.zeros(n)

epsilon = 0.01
iters_newton = 100
iters_grad = 1000

print("\n\nCORRIENDO MAIN")
print("Valor de la FO:", func_objetivo(x0, A, b))

print("\n ---------------- NEWTON ----------------")
newton(x0, A, b, epsilon, iters_newton)

print("\n ---------------- GRADIENTE ----------------")
gradiente(x0, A, b, epsilon, iters_grad)

print("\n ---------------- BFGS ----------------")
BFGS(x0, A, b, epsilon)