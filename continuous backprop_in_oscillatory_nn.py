import sympy as sym
import numpy as np
from matplotlib import pyplot as plt

num_nrns = 2
x = sym.Symbol("x")
a = sym.Symbol("a")
V = sym.MatrixSymbol("V", 1, num_nrns)
M = sym.MatrixSymbol("M", 1, num_nrns)
H = sym.MatrixSymbol("H", 1, num_nrns)
W_inh = sym.MatrixSymbol("W_inh", 1, num_nrns*num_nrns)
W_ex = sym.MatrixSymbol("W_ex", 1, num_nrns*num_nrns)
Z = sym.MatrixSymbol("Z", num_nrns, num_nrns*num_nrns) #parameter sensitivity

V_half = sym.Symbol("V_half") #
k_v = 4.0 #
k_ad = 0.9 #
C = 20 #

g_ad = 10.0 #
g_l = 2.8 #
g_SynE = 10.0 #
g_SynI = 60.0 #

E_K = -85.0 #
E_L = -60 #
E_SynE = 0.0 #
E_SynI = -75.0 #

tau_ad = 2000 #
drive = np.array([[0.5,0.55]]) #
# preparing rhs for the equations
# error-function
E = 0
for i in range(V.shape[1]):
    E += (V[0,i] - H[0,i])**2

#rhs of model equiations
f = np.zeros(num_nrns, dtype=object)
for i in range(num_nrns):
    Iad = g_ad * M[0,i] * (V[0,i] - E_K) #
    Il = g_l * (V[0,i] - E_L) #
    Itonic = g_SynE * (V[0,i] - E_SynE) * drive[0,i] #

    #adding synaptic currents
    IsynI = 0
    IsynE = 0
    for j in range(num_nrns):
        IsynI += g_SynI*W_inh[0, j*num_nrns + i]*(V[0, i] - E_SynI)*(1 / (1 + sym.exp(-(V[0, j] - V_half) / k_v)))
        IsynE += g_SynE*W_ex[ 0, j*num_nrns + i]*(V[0, i] - E_SynE)*(1 / (1 + sym.exp(-(V[0, j] - V_half) / k_v)))

    f[i] = - Iad - Il - Itonic - IsynE - IsynI
    f[i] = f[i]/C

M_eqs = np.zeros(num_nrns, dtype=object)
for i in range(num_nrns):
    M_eqs[i] = (k_ad * (1 / (1 + sym.exp(-(V[0,i] - V_half) / k_v))) - M[0,i]) / tau_ad


print("\nfunction F is defined : \n{}".format(f))
print("\nfunction M is defined : \n{}".format(M_eqs))


# f_diff = np.empty((num_nrns, num_nrns), dtype=object)
# for i in range(num_nrns):
#     for j in range(num_nrns):
#         f_diff[i, j] = f[i].diff(V[0,j])
# # print("\n the derivative of f is  : \n{}".format(f_diff))
#
# rhs_z = np.zeros((num_nrns,num_nrns*num_nrns), dtype=object)
# for i in range(num_nrns): # number of f rhs
#     for j in range(num_nrns*num_nrns): #number of parameter wrt to differentiate j -> (k,n)
#         for m in range(num_nrns):  # sweeping index
#             rhs_z[i,j] += f_diff[i,m]*Z[m,j]
# # print("\nz right hand side: \n {}".format(rhs_z))
#
# rhs_w = np.zeros((num_nrns*num_nrns,num_nrns), dtype=object)
# for i in range(num_nrns*num_nrns):
#     for j in range(num_nrns):
#         for k in range(num_nrns):
#             rhs_w[i,j] += sym.Abs(E.diff(V[0,j]))*Z[j,i]

# print("\nw right hand side: \n {}".format(rhs_w))

# LAMBDIFY
for i in range(f.shape[0]):
    f[i] = sym.lambdify((V,M,W_ex,W_inh,V_half),f[i],"numpy")
    M_eqs[i] = sym.lambdify((V,M,V_half),M_eqs[i],"numpy")

# test
# print(f[1](np.array([[10,20]]),np.array([[0,0.1,0.1,0]])))

# for i in range(rhs_z.shape[0]):
#     for j in range(rhs_z.shape[1]):
#         rhs_z[i,j] = sym.lambdify((V,W_inh,Z),rhs_z[i,j],"numpy")
# # test
# # print(rhs_z[0,1](np.random.rand(1,num_nrns),np.random.rand(1,num_nrns*num_nrns),np.random.rand(num_nrns,num_nrns*num_nrns) ) )
#
# for i in range(rhs_w.shape[0]):
#     for j in range(rhs_w.shape[1]):
#         rhs_w[i,j] = sym.lambdify((V,H,W_inh,Z),rhs_w[i,j],"numpy")
#
# # test
# # print(rhs_w[0,1](np.random.rand(1,num_nrns),np.random.rand(1,num_nrns),np.random.rand(1,num_nrns*num_nrns),np.random.rand(num_nrns,num_nrns*num_nrns) ) )


#generating the target signal with those equations
V = -70 + 40*np.random.rand(1,2)
M = np.random.rand(1,2)
V_half = -30
W_inh = np.array([[0, -1.4, -0.9, 0]]) #np.array([[0, 0.8, 0.8, 0]]) ???
W_ex = np.array([[0, 0.0, 0.0, 0]])
dt = 0.1
vals_V = [[V[0,0]],[V[0,1]]]
vals_M = [[M[0,0]],[M[0,1]]]
t = [0]
stoptime = 20000
for i in range(int(stoptime/dt)):
    for j in range(len(M_eqs)):
        M[0, j] = M[0, j] + dt*M_eqs[j](V, M, V_half)
        vals_M[j].append(M[0, j])

    for j in range(len(f)):
        V[0,j] = V[0,j] + dt*f[j](V, M, W_inh, W_ex,V_half)
        vals_V[j].append(V[0,j])
    t.append(t[-1] + dt)


fr1 = 1/(1+np.exp(-(np.array(vals_V[0]) - V_half)/k_v))
fr2 = 1/(1+np.exp(-(np.array(vals_V[1]) - V_half)/k_v))

startime = 5000
start = int(startime/dt)
fig = plt.figure(figsize=(10,4))
plt.plot(t[start:], fr1[start:], "k--", linewidth = 3, alpha = 0.7)
plt.plot(t[start:], fr2[start:], "r-", linewidth = 3, alpha = 0.7)
plt.grid(True)
# plt.ylim([-1,1])
plt.show()
