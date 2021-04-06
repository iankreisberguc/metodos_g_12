import numpy as np
import numdifftools as nd
from parametros import *

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


# Calcula el gradiente de la FO de manera analitica
#def gradient_f(x, A, b, a=constant):
#    return np.dot(aux_vector, A)
#    aux_vector = a**2 * 2 * residuo(x, A, b) / (a**2 - residuo(x, A, b)**2)

# Calcula el hessiano de la FO de manera numerica
#def hessian (x, A, b):
    
    # ------------------------------
    # Calculo usando diferencias finitas
    # ------------------------------
    #N = x.shape[0]
    #hessian = np.zeros((N, N)) 
    #gd_0 = gradient_f(x, A, b)
    #eps = np.linalg.norm(gd_0) * np.finfo(np.float32).eps 
    #for i in range(N):
    #    xx0 = 1. * x[i]
    #    x[i] = xx0 + eps
    #    gd_1 =  gradient_f(x, A, b)
    #    hessian[:, i] = ((gd_1 - gd_0) / eps)
    #    x[i] = xx0
    #return hessian

    # ------------------------------
    # Calculo usando el hessiano de nd sobre la funcion objetivo
    # ------------------------------
    #hess = nd.Hessian(func_objetivo)
    #return hess(x, A, b)

    # ------------------------------
    # Calculo usando el gradiende de nd sobre el gradiente de la funcion objetivo
    # ------------------------------
    #hess = nd.Gradient(gradient_f)
    #return hess(x, A, b


# Testeo de BFGS, primero se generan datos para la funcion
m = 300
n = 50
    
#    np.random.seed(1610)
np.random.seed(220399)
    
A, b = generar_datos(m, n)
residuo = lambda x, A, b: np.dot(A, x) - b
x_opt = [-0.00090309,  0.00118256, -0.00100717, -0.00056781, -0.00205902,  0.00368465,
 -0.0022257,   0.00025377, -0.00182315,  0.00117174,  0.00123833,  0.00172285,
  0.00424657, -0.00023223,  0.0037284,   0.00054259, -0.00204912,  0.00091248,
  0.00335923, -0.00336994,  0.00051575, -0.00194586,  0.00196556, -0.00144571,
  0.00215869,  0.00218757, -0.00121628, -0.00055436, -0.00313988,  0.00229273,
 -0.00149506,  0.00287618, -0.00169493, -0.00168946, -0.00079053,  0.00655203,
  0.0007888,  -0.0018132,  -0.00192264,  0.0014942,   0.00222382,  0.00028745,
  0.00208139, -0.00229559, -0.00244832, -0.00284838,  0.00101864, -0.00155116,
 -0.00158425, -0.00217585]
x_opt = np.array(x_opt)
print(np.linalg.norm(residuo(x_opt, A, b)))