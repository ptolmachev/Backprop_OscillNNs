import sympy as sym
import numpy as np


def get_RHS(params):
    num_nrns = params["num_nrns"]

    k_v = params["k_v"]
    k_ad = params["k_ad"]
    C = params["C"]
    g_ad = params["g_ad"]
    g_l = params["g_l"]
    g_SynE = params["g_SynE"]
    g_SynI = params["g_SynI"]
    E_K = params["E_K"]
    E_L = params["E_L"]
    E_SynE = params["E_SynE"]
    E_SynI = params["E_SynI"]
    tau_ad = params["tau_ad"]
    drive = params["drive"]

    # SYMBOLIC VARIABLES
    V = sym.MatrixSymbol("V", 1, num_nrns)
    H = sym.MatrixSymbol("H", 1, num_nrns)
    M = sym.MatrixSymbol("M", 1, num_nrns)
    Z_inh = sym.MatrixSymbol("Z_inh", num_nrns, num_nrns*num_nrns) # V sensitivity to W_inh
    Y_inh = sym.MatrixSymbol("Y_inh", num_nrns, num_nrns*num_nrns) # M sensitivity to W_inh
    W_inh = sym.MatrixSymbol("W_inh", 1, num_nrns*num_nrns)
    V_half = sym.Symbol("V_half")


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

    # defining rhs of '''dm/dt = G(m,v)''' symbolically
    G = np.zeros(num_nrns, dtype=object)
    for i in range(num_nrns):
        G[i] = (k_ad * (1 / (1 + sym.exp(-(V[0,i] - V_half) / k_v))) - M[0,i]) / tau_ad

    #DIFFERENTIATING F WRT V
    F_diff_v = np.empty((num_nrns, num_nrns), dtype=object)
    for i in range(num_nrns):
        for j in range(num_nrns):
            F_diff_v[i, j] = F[i].diff(V[0,j])
    # print("\n the derivative of Fm is  : \n{}".format(F_diff_v))

    #DIFFERENTIATING F WRT M
    F_diff_m = np.empty((num_nrns, num_nrns), dtype=object)
    for i in range(num_nrns):
        for j in range(num_nrns):
            F_diff_m[i, j] = F[i].diff(M[0,j])
    # print("\n the derivative of Fm is  : \n{}".format(F_diff_m))

    #DIFFERENTIATING F WRT W
    F_diff_w = np.empty((num_nrns, num_nrns*num_nrns), dtype=object)
    for i in range(num_nrns):
        for j in range(num_nrns*num_nrns):
            F_diff_w[i, j] = F[i].diff(W_inh[0,j])
    # print("\n the derivative of Fm is  : \n{}".format(F_diff_w))

    #DIFFERENTIATING G WRT M
    G_diff_m = np.empty((num_nrns, num_nrns), dtype=object)
    for i in range(num_nrns):
        for j in range(num_nrns):
            G_diff_m[i, j] = G[i].diff(M[0,j])
    # print("\n the derivative of Gm is  : \n{}".format(G_diff_m))

    # DEFINING RHS FOR Z and Y-VARIABLES
    RHS_Z_inh = np.zeros((num_nrns,num_nrns*num_nrns), dtype=object)
    RHS_Y_inh = np.zeros((num_nrns,num_nrns*num_nrns), dtype=object)
    for i in range(num_nrns): # number of f rhs
        for j in range(num_nrns*num_nrns): #number of parameter wrt to differentiate j -> (k,n)
            for m in range(num_nrns):  # sweeping index
                RHS_Z_inh[i, j] +=  F_diff_v[i, m] * Z_inh[m, j] + F_diff_m[i, m] * Y_inh[m, j]
                RHS_Y_inh[i, j] +=  G_diff_m[i, m] * Y_inh[m, j]
            RHS_Z_inh[i, j] += F_diff_w[i, j]

    # ERROR FUNCTION
    E = 0
    for i in range(V.shape[1]):
        E += (V[0,i] - H[0,i])**2

    # DEFINING RHS FOR W-parameters
    RHS_W_inh = np.zeros(num_nrns*num_nrns, dtype=object)
    for j in range(num_nrns*num_nrns):
        for k in range(num_nrns):
            RHS_W_inh[j] -= ( (E.diff(V[0, k])) * Z_inh[k, j]  ) #+ (E.diff(M[0, k])) * Y_inh[k, j]

    # CASTING SYMBOLIC FUNCTION INTO NUMPY (LAMBDIFY)
    for i in range(F.shape[0]):
        F[i] = F[i].subs(V,H)
        G[i] = G[i].subs(V,H)
        F[i] = sym.lambdify((H, M, W_inh, V_half),F[i],"numpy")
        G[i] = sym.lambdify((H, M, V_half),G[i],"numpy")

    for i in range(RHS_Z_inh.shape[0]):
        for j in range(RHS_Z_inh.shape[1]):
            RHS_Z_inh[i, j] = RHS_Z_inh[i, j].subs(V,H)
            RHS_Z_inh[i, j] = sym.lambdify((H, M, W_inh, Z_inh, Y_inh, V_half), RHS_Z_inh[i, j], "numpy")

    for i in range(RHS_Y_inh.shape[0]):
        for j in range(RHS_Y_inh.shape[1]):
            RHS_Y_inh[i, j] = RHS_Y_inh[i, j].subs(V,H)
            RHS_Y_inh[i, j] = sym.lambdify((H, M, W_inh, Z_inh, Y_inh, V_half), RHS_Y_inh[i, j], "numpy")

    for j in range(RHS_W_inh.shape[0]):
        RHS_W_inh[j] = sym.lambdify((V, H, Z_inh, Y_inh), RHS_W_inh[j], "numpy")

    return F, G, RHS_Z_inh, RHS_Y_inh, RHS_W_inh