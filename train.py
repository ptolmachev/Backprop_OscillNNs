import numpy as np

def train(RB, num_iter, num_nrns, etha, V_half, dt, RHS):
    decay = np.exp(np.log(0.4*etha)/num_iter)
    print("decay is : ", decay)
    F, G, RHS_Z_inh, RHS_Y_inh, RHS_W_inh = RHS

    V = -70 + 40 * np.random.rand(1, num_nrns)

    #random initial guess
    W_inh = 2*np.random.rand(1, num_nrns**2)
    W_inh[0, 0] = 0.0
    W_inh[0, 4] = 0.0
    W_inh[0, 8] = 0.0
    # W_inh = np.array([[0.0, 0.1, 0.1, 0.0]])
    Z_inh = 0.0 * np.ones((num_nrns, num_nrns * num_nrns))  ###!
    Y_inh = 0.0 * np.ones((num_nrns, num_nrns * num_nrns))  ###!

    vals_Z_inh = np.empty((num_nrns, num_nrns * num_nrns), dtype=list)
    for i in range(vals_Z_inh.shape[0]):
        for j in range(vals_Z_inh.shape[1]):
            vals_Z_inh[i, j] = []
            vals_Z_inh[i, j].append(Z_inh[i, j])

    vals_Y_inh = np.empty((num_nrns, num_nrns * num_nrns), dtype=list)
    for i in range(vals_Y_inh.shape[0]):
        for j in range(vals_Y_inh.shape[1]):
            vals_Y_inh[i, j] = []
            vals_Y_inh[i, j].append(Y_inh[i, j])


    vals_W_inh = np.empty((1, num_nrns * num_nrns), dtype=list)
    for i in range(vals_W_inh.shape[1]):
        vals_W_inh[0, i] = []
        vals_W_inh[0, i].append(W_inh[0, i])

    for i in range(num_iter):
        M, H, H_target = RB.sample()
        if (i % 1000 == 0):
            print("\rStep number {} ".format(i))

        #update Z, Y
        for j in range(Z_inh.shape[0]):
            for k in range(Z_inh.shape[1]):
                Z_inh[j, k] = 0.99*Z_inh[j, k] + np.clip(dt * (RHS_Z_inh[j, k](H, M, W_inh, Z_inh, Y_inh, V_half)), -5, 5) #!!
                vals_Z_inh[j, k].append(Z_inh[j, k])

        for j in range(Y_inh.shape[0]):
            for k in range(Y_inh.shape[1]):
                Y_inh[j, k] = Y_inh[j, k] + np.clip(dt * RHS_Y_inh[j, k](H, M, W_inh, Z_inh, Y_inh, V_half), -5, 5)
                vals_Y_inh[j, k].append(Y_inh[j, k])

        #predict V
        for j in range(len(F)):
            V[0, j] = H[0, j] + dt * F[j](H, M, W_inh, V_half)

        #update W
        for k in range(W_inh.shape[1]):
            if (k == 0) or (k == 4) or (k==8):# or (k == 0) or (k == 3):
                W_inh[0, k] = W_inh[0, k]
                vals_W_inh[0, k].append(W_inh[0, k])
            else:
                # W_inh[0, k] = W_inh[0, k] + etha*np.clip(dt*RHS_W_inh[k](V, H, Z_inh, Y_inh),-1000,1000)
                W_inh[0, k] = np.clip(W_inh[0, k] + etha * dt * RHS_W_inh[k](V, H_target, Z_inh, Y_inh), 0, 1)
                # vals_test[0, k].append(np.clip(dt*RHS_W_inh[k](V, H, Z_inh, Y_inh),-1000,1000))
                vals_W_inh[0, k].append(W_inh[0, k])

        etha *= decay

    return vals_W_inh, vals_Z_inh, vals_Y_inh
