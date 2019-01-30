import numpy as np


def get_experiences(ReplayBuffer, num_iter, num_nrns, dt, threshold, vals_V_target, RHS):
    F, G, RHS_Z_inh, RHS_Y_inh, RHS_W_inh = RHS

    # INITIAL CONDITIONS
    M = np.random.rand(1, num_nrns)
    V_half = -30

    # EXPERIENCE LOOP
    for i in range(num_iter):
        ten_perc = int(0.1*num_iter)
        if (i % ten_perc == 0):
            print("\rSimulated {}%".format(10 * i / ten_perc))

        H = np.random.rand(1,num_nrns)
        for j in range(num_nrns):
            H[0,j] = vals_V_target[j][i]

        for j in range(len(G)):
            M[0, j] = M[0, j] + dt*G[j](H, M, V_half)

        H_target = np.random.rand(1,num_nrns)
        for j in range(num_nrns):
            H_target[0,j] = vals_V_target[j][i+1]

        if int(i/dt) > threshold:
            experience = (M, H, H_target)
            ReplayBuffer.memorize(experience)
