import sympy as sym
import numpy as np
from matplotlib import pyplot as plt
import pickle

num_nrns = 2
# SYMBOLIC VARIABLES
x = sym.Symbol("x")
V = sym.MatrixSymbol("V", 1, num_nrns)
M = sym.MatrixSymbol("M", 1, num_nrns)
W_inh = sym.MatrixSymbol("W_inh", 1, num_nrns*num_nrns)

# MODEL PARAMETERS
V_half = sym.Symbol("V_half")
k_v = 4.0
k_ad = 0.9
C = 20
g_ad = 10.0
g_l = 2.8
g_SynE = 10.0
g_SynI = 60.0
E_K = -85.0
E_L = -60
E_SynE = 0.0
E_SynI = -75.0
tau_ad = 2000
drive = np.array([[0.5,0.55]])

# PREPARING THE EQUATIONS
# defining rhs of '''dv/dt = F(m,v,w)''' symbolically
F = np.zeros(num_nrns, dtype=object)
for i in range(num_nrns):
    Iad = g_ad * M[0,i] * (V[0,i] - E_K) #
    Il = g_l * (V[0,i] - E_L) #
    Itonic = g_SynE * (V[0,i] - E_SynE) * drive[0,i] #

    #adding synaptic currents
    IsynI = 0
    for j in range(num_nrns):
        IsynI += g_SynI*W_inh[0, j*num_nrns + i]*(V[0, i] - E_SynI)*(1 / (1 + sym.exp(-(V[0, j] - V_half) / k_v)))

    F[i] = - Iad - Il - Itonic - IsynI
    F[i] = F[i]/C

# defining rhs of '''dm/dt = M(m,v)''' symbolically
M_eqs = np.zeros(num_nrns, dtype=object)
for i in range(num_nrns):
    M_eqs[i] = (k_ad * (1 / (1 + sym.exp(-(V[0,i] - V_half) / k_v))) - M[0,i]) / tau_ad


print("\nfunction F is defined : \n{}".format(F))
print("\nfunction M is defined : \n{}".format(M_eqs))


#DIFFERENTIATING F WRT V
F_diff = np.empty((num_nrns, num_nrns), dtype=object)
for i in range(num_nrns):
    for j in range(num_nrns):
        F_diff[i, j] = F[i].diff(V[0,j])
print("\n the derivative of f is  : \n{}".format(F_diff))

# DEFINING RHS FOR Z-VARIABLES
Z_inh = sym.MatrixSymbol("Z_inh", num_nrns, num_nrns*num_nrns) # parameter sensitivity
RHS_Z_inh = np.zeros((num_nrns,num_nrns*num_nrns), dtype=object)
for i in range(num_nrns): # number of f rhs
    for j in range(num_nrns*num_nrns): #number of parameter wrt to differentiate j -> (k,n)
        for m in range(num_nrns):  # sweeping index
            RHS_Z_inh[i, j] += F_diff[i, m] * Z_inh[m, j]

# ERROR FUNCTION
H = sym.MatrixSymbol("H", 1, num_nrns)
E = 0
for i in range(V.shape[1]):
    E += (V[0,i] - H[0,i])**2

# DEFINING RHS FOR W-parameters
RHS_W_inh = np.zeros(num_nrns*num_nrns, dtype=object)
for j in range(num_nrns*num_nrns):
    for k in range(num_nrns):
        RHS_W_inh[j] -= (E.diff(V[0, k])) * Z_inh[k, j]

# CASTING SYMBOLIC FUNCTION INTO NUMPY (LAMBDIFY)
for i in range(F.shape[0]):
    F[i] = sym.lambdify((V,M,W_inh,V_half),F[i],"numpy")
    M_eqs[i] = sym.lambdify((V,M,V_half),M_eqs[i],"numpy")

for i in range(RHS_Z_inh.shape[0]):
    for j in range(RHS_Z_inh.shape[1]):
        RHS_Z_inh[i, j] = sym.lambdify((V, M, W_inh, Z_inh, V_half), RHS_Z_inh[i, j], "numpy")

for j in range(RHS_W_inh.shape[0]):
    RHS_W_inh[j] = sym.lambdify((V, H, Z_inh), RHS_W_inh[j], "numpy")

# LOADING THE TARGET SIGNAL
data = pickle.load(open("signal_target.dat", "rb+"))
vals_V_target = data["vals_V"]
t_target = data["t"]

V = -70 + 40*np.random.rand(1,2)
M = np.random.rand(1,2)
Z_inh = np.random.rand(num_nrns, num_nrns*num_nrns)
V_half = -30
W_inh = np.array([[0, 0.2, 0.2, 0]])
dt = 0.1
vals_V = [[V[0,0]],[V[0,1]]]
vals_M = [[M[0,0]],[M[0,1]]]

vals_Z_inh = np.empty((num_nrns, num_nrns*num_nrns), dtype=list)
for i in range(vals_Z_inh.shape[0]):
    for j in range(vals_Z_inh.shape[1]):
        vals_Z_inh[i, j] = []
        vals_Z_inh[i,j].append(Z_inh[i,j])

vals_W_inh = np.empty((1, num_nrns*num_nrns), dtype=list)
for i in range(vals_W_inh.shape[1]):
    vals_W_inh[0, i] = []
    vals_W_inh[0, i].append(W_inh[0, i])

t = [0]
stoptime = t_target[-1]
for i in range(int(stoptime/dt)):

    if ( int(i/dt)%int(0.1*stoptime/dt)== 0):
        print("\rSimulated {}%".format(np.ceil(i*100/int(stoptime/dt))))

    for j in range(len(M_eqs)):
        M[0, j] = M[0, j] + dt*M_eqs[j](V, M, V_half)
        vals_M[j].append(M[0, j])

    for j in range(len(F)):
        V[0,j] = V[0,j] + dt*F[j](V, M, W_inh, V_half)
        vals_V[j].append(V[0,j])

    for j in range(Z_inh.shape[0]):
        for k in range(Z_inh.shape[1]):
            Z_inh[j,k] = Z_inh[j,k] + dt*RHS_Z_inh[j,k](V, M, W_inh, Z_inh, V_half)
            vals_Z_inh[j,k].append(Z_inh[j,k])

    H = np.random.rand(1,2)
    H[0,0] = vals_V_target[0][i]
    H[0,1] = vals_V_target[1][i]

    for j in range(W_inh.shape[1]):
        if j == 1 : #changing only one of parameters
            W_inh[0, j] = W_inh[0, j] + 0.01*dt*RHS_W_inh[j](V, H, Z_inh)
            vals_W_inh[0, j].append(W_inh[0, j])
    t.append(t[-1] + dt)

# PLOTTING THE RESULTS
fr1 = 1/(1+np.exp(-(np.array(vals_V[0]) - V_half)/k_v))
fr2 = 1/(1+np.exp(-(np.array(vals_V[1]) - V_half)/k_v))

startime = 1
# start = int(startime/dt)
# fig = plt.figure(figsize=(10,4))
# plt.plot(t[start:], fr1[start:], "k--", linewidth = 3, alpha = 0.7)
# plt.plot(t[start:], fr2[start:], "r-", linewidth = 3, alpha = 0.7)
# plt.grid(True)
# plt.show()

print(W_inh)
fig = plt.figure(figsize=(10,4))
plt.plot(t,vals_W_inh[0, 1], "k--", linewidth = 3, alpha = 0.7)
plt.grid(True)
plt.show()
