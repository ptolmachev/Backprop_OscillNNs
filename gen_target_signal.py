import sympy as sym
import numpy as np
from matplotlib import pyplot as plt
import pickle

num_nrns = 2
# SYMBOLIC VARIABLES
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

    F[i] = -Iad - Il - Itonic - IsynI
    F[i] = F[i]/C

# defining rhs of '''dm/dt = M(m,v)''' symbolically
M_eqs = np.zeros(num_nrns, dtype=object)
for i in range(num_nrns):
    M_eqs[i] = (k_ad * (1 / (1 + sym.exp(-(V[0,i] - V_half) / k_v))) - M[0,i]) / tau_ad

print("\nfunction F is defined : \n{}".format(F))
print("\nfunction M is defined : \n{}".format(M_eqs))

# CASTING SYMBOLIC FUNCTION INTO NUMPY (LAMBDIFY)
for i in range(F.shape[0]):
    F[i] = sym.lambdify((V,M,W_inh,V_half),F[i],"numpy")
    M_eqs[i] = sym.lambdify((V,M,V_half),M_eqs[i],"numpy")


# GENERATING THE TARGET SIGNAL
V = -70 + 40*np.random.rand(1,2)
M = np.random.rand(1,2)
V_half = -30
W_inh = np.array([[0, 0.4, 0.2, 0]])
dt = 0.1
vals_V = [[V[0,0]],[V[0,1]]]
vals_M = [[M[0,0]],[M[0,1]]]
t = [0]
stoptime = 20000

for i in range(int(stoptime/dt)):

    for j in range(len(M_eqs)):
        M[0, j] = M[0, j] + dt*M_eqs[j](V, M, V_half)
        vals_M[j].append(M[0, j])

    for j in range(len(F)):
        V[0,j] = V[0,j] + dt*F[j](V, M, W_inh, V_half)
        vals_V[j].append(V[0,j])

    t.append(t[-1] + dt)

# PLOTTING THE RESULTS
fr1 = 1/(1+np.exp(-(np.array(vals_V[0]) - V_half)/k_v))
fr2 = 1/(1+np.exp(-(np.array(vals_V[1]) - V_half)/k_v))

startime = 3000
start = int(startime/dt)
fig = plt.figure(figsize=(10,4))
plt.plot(t[start:], fr1[start:], "k--", linewidth = 3, alpha = 0.7)
plt.plot(t[start:], fr2[start:], "r-", linewidth = 3, alpha = 0.7)
plt.grid(True)
plt.show()

# SAVING TARGET SIGNAL
file = open("signal_target.dat","wb+")
data = dict()
data["vals_V"] = vals_V
data["vals_M"] = vals_M
data["t"] = t
data["V_half"] = V_half
data["k_v"] = k_v
pickle.dump(data, file)