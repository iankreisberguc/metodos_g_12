import numpy as np
import numdifftools as nd


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