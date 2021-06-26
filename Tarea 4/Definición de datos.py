from gurobipy import GRB, Model, quicksum
import numpy as np
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:40:48 2021

@author: jrver
"""

# Acá se definen los datos, con la notacuión de enunciado
m = 12 # plantas de proceso
n = 140 # parcelas
hor = 12
M = range(m)
N = range(n)
T = range(hor)

np.random.seed(71099)

c = np.random.randint(150,400,size=(n,hor)) # costo por kilo cosechado
f = np.random.randint(200000,300000,size=(n,hor)) # costo fijo por cosechar una parcela
p = np.random.randint(800,1000,size=m) # precio dependiendo de la planta en donde se proceso
e = np.random.randint(10,15,size=(m,hor)) # costo de procesarlo
B = np.random.randint(n*10,n*30,size=hor) # Mano de obra disponible
K = np.random.randint(n*700,n*1000,size=m) # Kilos que puede recibir un planta
alpha = np.random.randint(7,10,size=(n,hor))/50 # requerimiento de mano de obra
L = np.random.randint(1500,2000,size=n) # disponibilidad de fruta en cada parcela




# Generación del modelo
model = Model("Cosechas")

# Crear y rellenar diccionarios de variables manufactura b_t,p , almacenada s_t,p , vendida u_t,p
X_jt = {} # kilos cosechados en la parcela j el dia t
Y_jt = {} # vale 1 si se cosecha la parcela j en el día t, 0 en caso contrario
Z_it = {} # kilos procesados en la planta i en el periodo t

for t in T:
    for j in N:
        X_jt[j, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="X_{}_{}".format(j, t))
        Y_jt[j, t] = model.addVar(vtype=GRB.BINARY, name="Y_{}_{}".format(j, t))
    for i in M:
        Z_it[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Z_{}_{}".format(i, t))


# Llama a update para agregar las variables al modelo
model.update()
         
# Restriccion 1

for t in T:
    model.addConstr(quicksum(alpha[j][t] * X_jt[j, t] for j in N) <= B[t])

# Restriccion 2
for t in T:
    for j in N:
        model.addConstr(X_jt[j, t] <= L[j]* Y_jt[j, t])

# Restriccion 3
for j in N:
    model.addConstr(quicksum(Y_jt[j, t] for t in T) <= 1 )

# Restriccion 4
for t in T:
    model.addConstr(quicksum(X_jt[j, t] for j in N) == quicksum(Z_it[i, t] for i in M))

# Restriccion 5
for i in M:
    model.addConstr(quicksum(Z_it[i, t] for t in T) <= K[i])

# Funcion objetivo

obj= quicksum(quicksum(p[i]*Z_it[i, t] for i in M) for t in T) -\
     quicksum(
         (
            quicksum(c[j][t]*X_jt[j, t] for j in N) +\
            quicksum(f[j][t]*Y_jt[j, t] for j in N) + \
            quicksum(e[i][t]*Z_it[i, t] for i in M)
         ) for t in T
     )

model.setObjective(obj, GRB.MAXIMIZE)

model.optimize()

# Mostrar los valores de las soluciones
model.printAttr("X")


