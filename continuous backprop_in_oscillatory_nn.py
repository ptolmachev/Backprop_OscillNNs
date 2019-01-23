import sympy as sym
import numpy as np
from matplotlib import pyplot as plt

num_nrns = 2
x = sym.Symbol("x")
a = sym.Symbol("a")
V = sym.MatrixSymbol("V", 1, num_nrns)
M = sym.MatrixSymbol("V", 1, num_nrns)
H = sym.MatrixSymbol("H", 1, num_nrns)
W = sym.MatrixSymbol("W", 1, num_nrns*num_nrns)
Z = sym.MatrixSymbol("Z", num_nrns, num_nrns*num_nrns) #parameter sensitivity
x_0 = -45
k = 8.0
tau = 100
# #preparing rhs for the equations
# #error-function
E = 0
for i in range(V.shape[1]):
    E += (V[0,i] - H[0,i])**2

#rhs of model equiations
f = np.zeros(num_nrns, dtype=object)
for i in range(num_nrns):
    f[i] = -V[0,i]
    for j in range(num_nrns):
        f[i] += W[0,j*num_nrns + i]* (1/(1+sym.exp(-(V[0,j] - x_0)/k)))
    f[i] = f[i]/tau

print("\nfunction f is defined : \n{}".format(f))


f_diff = np.empty((num_nrns, num_nrns), dtype=object)
for i in range(num_nrns):
    for j in range(num_nrns):
        f_diff[i, j] = f[i].diff(V[0,j])
# print("\n the derivative of f is  : \n{}".format(f_diff))

rhs_z = np.zeros((num_nrns,num_nrns*num_nrns), dtype=object)
for i in range(num_nrns): # number of f rhs
    for j in range(num_nrns*num_nrns): #number of parameter wrt to differentiate j -> (k,n)
        for m in range(num_nrns):  # sweeping index
            rhs_z[i,j] += f_diff[i,m]*Z[m,j]
# print("\nz right hand side: \n {}".format(rhs_z))

rhs_w = np.zeros((num_nrns*num_nrns,num_nrns), dtype=object)
for i in range(num_nrns*num_nrns):
    for j in range(num_nrns):
        for k in range(num_nrns):
            rhs_w[i,j] += sym.Abs(E.diff(V[0,j]))*Z[j,i]

# print("\nw right hand side: \n {}".format(rhs_w))

# LAMBDIFY
for i in range(f.shape[0]):
    f[i] = sym.lambdify((V,W),f[i],"numpy")

# test
# print(f[1](np.array([[10,20]]),np.array([[0,0.1,0.1,0]])))

for i in range(rhs_z.shape[0]):
    for j in range(rhs_z.shape[1]):
        rhs_z[i,j] = sym.lambdify((V,W,Z),rhs_z[i,j],"numpy")
# test
# print(rhs_z[0,1](np.random.rand(1,num_nrns),np.random.rand(1,num_nrns*num_nrns),np.random.rand(num_nrns,num_nrns*num_nrns) ) )

for i in range(rhs_w.shape[0]):
    for j in range(rhs_w.shape[1]):
        rhs_w[i,j] = sym.lambdify((V,H,W,Z),rhs_w[i,j],"numpy")

# test
# print(rhs_w[0,1](np.random.rand(1,num_nrns),np.random.rand(1,num_nrns),np.random.rand(1,num_nrns*num_nrns),np.random.rand(num_nrns,num_nrns*num_nrns) ) )


#generating the target signal with those equations
V = -60 +20*np.random.rand(1,2)
W = np.array([[0,-1,-1, 0]])
dt = 0.1
vals = [[V[0,0]],[V[0,1]]]
t = [0]
stoptime = 10000
for i in range(int(stoptime/dt)):
    for j in range(len(f)):
        V[0,j] = V[0,j] + dt*f[j](V,W)
        vals[j].append(V[0,j])
    t.append(t[-1] + dt)


fr1 = 1/(1+np.exp(-(np.array(vals[0]) - x_0)/k))
fr2 = 1/(1+np.exp(-(np.array(vals[1]) - x_0)/k))

fig = plt.figure(figsize=(10,4))
plt.plot(t, fr1, "k--", linewidth = 3, alpha = 0.7)
plt.plot(t, fr2, "r-", linewidth = 3, alpha = 0.7)
plt.grid(True)
# plt.ylim([-1,1])
plt.show()
