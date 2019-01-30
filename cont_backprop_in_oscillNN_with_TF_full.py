import numpy as np
from matplotlib import pyplot as plt
import pickle
from ReplayBuffer import ReplayBuffer
from RHS import get_RHS
from get_experiences import get_experiences
from train import train

# def unsquash(fr, V_half, k_v):
#     return V_half - k_v*np.log(1.0/np.array(fr) - 1)

num_nrns = 3
etha = 2*1e-3

# REPLAY BUFFER
params = dict()
params["maxlen"] = 1000000
params["batch_size"] = 100
RB = ReplayBuffer(params)

# LOADING THE TARGET SIGNAL
data = pickle.load(open("signal_target.dat", "rb+"))
V_half = data["V_half"]
k_v = data["k_v"]
vals_fr_target = data["vals_fr"]
# vals_V_target = unsquash(vals_fr_target,V_half, k_v)
vals_V_target = data["vals_V"]
vals_M_target = data["vals_M"]
t_target = data["t"]
dt = data["dt"]
true_vals = data["true_weights"][0]

# MODEL PARAMETERS
params = dict()
params["k_v"] = 4.0
params["k_ad"] = 0.9
params["C"] = 20
params["g_ad"] = 10.0
params["g_l"] = 2.8
params["g_SynE"] = 10.0
params["g_SynI"] = 60.0
params["E_K"] = -85.0
params["E_L"] = -60
params["E_SynE"] = 0.0
params["E_SynI"] = -75.0
params["tau_ad"] = 2000
params["drive"] = np.array([[0.5,0.55,0.5]])
params["num_nrns"] = num_nrns

F, G, RHS_Z_inh, RHS_Y_inh, RHS_W_inh = get_RHS(params)
RHS = F, G, RHS_Z_inh, RHS_Y_inh, RHS_W_inh


stoptime = t_target[-1]
num_iter = int(stoptime/dt)-1

threshold = 15000 #memorize experiences only after "threshold" time

get_experiences(RB, num_iter, num_nrns, dt, threshold, vals_V_target, RHS)

num_iter = 60000
vals_W_inh, vals_Z_inh, vals_Y_inh = train(RB, num_iter, num_nrns, etha, V_half, dt, RHS)

# PLOTTING THE RESULTS
fig1 = plt.figure(figsize=(10,4))
styles = ["k","b","r","g","c", "m", "y", "w", "#eeefff"]
for ind, i in enumerate([i for i in range(num_nrns**2) if (i % num_nrns) != (int(i/num_nrns)) ]):
    plt.plot(vals_W_inh[0, ind], styles[i], linewidth = 3, alpha = 0.7)
    plt.plot(true_vals[ind]*np.ones(len(vals_W_inh[0, ind])), styles[i] + str("--"), linewidth=3, alpha=0.3)
plt.grid(True)
fig1.suptitle('evolution of the connectivity parameters', fontsize=16)
plt.show()

fig2 = plt.figure(figsize=(10,4))
for ind, i in enumerate([i for i in range(num_nrns**2) if (i % num_nrns) != (int(i/num_nrns)) ]):
    plt.plot(vals_Z_inh[0, ind], styles[i], linewidth = 3, alpha = 0.7)
plt.grid(True)
fig2.suptitle('Sensitivity coefficients 1st neuron', fontsize=16)
plt.show()

print("\nThe resulting weights are: \n")
for i in range(num_nrns**2):
    print(np.median(vals_W_inh[0, i][-100:]))
