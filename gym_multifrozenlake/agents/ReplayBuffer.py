from ast import Param


class ReplayBuffer(object):

    def __init__(self, size):
        self.size = size

    # value computed by critic
    
    def storeTransition(self, state, action, reward, value, terminated, logProb):
        print("Not implemented")