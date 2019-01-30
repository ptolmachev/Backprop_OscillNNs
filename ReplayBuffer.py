from collections import deque
import numpy as np

class ReplayBuffer():
    def __init__(self, params):
        self.maxlen = params["maxlen"]
        # self.batch_size = params["batch_size"]
        self.memory = deque(maxlen = self.maxlen)

    def sample(self):
        ind = np.random.randint(len(self.memory))
        return list(self.memory)[ind]

    def get_next(self):
        return self.memory.pop()

    def memorize(self, experience):
        #experience contains M, Z, Y, W, H, H_target
        self.memory.append(experience)

# test
# params = dict()
# params["maxlen"] = 50
# params["batch_size"] = 25
# RB = ReplayBuffer(params)
#
# for i in range(100):
#     RB.memorize((np.random.randint(100),np.random.randint(100)))
#
# print(RB.sample())