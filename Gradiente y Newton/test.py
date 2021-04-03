import numpy as np


np.random.seed(3)
r_i = lambda A, x, b: np.dot(A, x) - b
A = np.random.rand(5, 5)
a = 1.1
x = np.random.rand(5)
b = np.ones(5)
funcion_objetivo = lambda x, a: np.piecewise(x, [(abs(x) < a), (abs(x) >= a)], 
                                            [(-a ** 2) * np.log(1 - (x / a) ** 2), 2**62])
#print(A)
#print(x)
#print(b)
#print(r_i(A, x, b))
#ri = r_i(A, x, b)
x = 10
a = 1
print(funcion_objetivo(x, a))