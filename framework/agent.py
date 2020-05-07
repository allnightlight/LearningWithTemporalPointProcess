

import torch 
import tensorflow as tf
import torch.nn as nn 


# <<interface>>
class IAgent:

    def save(self, saveFilePath):
        raise NotImplementedError

    def load(self, saveFilePath):
        raise NotImplementedError


class TestAgent(IAgent):

    def save(self, saveFilePath):
        pass

    def load(self, saveFilePath):
        pass


# <<abstract>>
class AgentTf(tf.keras.Model, IAgent):

    def __init__(self):
        super().__init__()

    def save(self, saveFilePath):
        self.save_weights(saveFilePath)

    def load(self, saveFilePath):
        self.load_weights(saveFilePath)

# <<abstract>>
    def call(self, *arg):
        raise NotImplementedError()


# <<abstract>>
class AgentTorch(nn.Module, IAgent):

    def __init__(self):
        super().__init__()

    def save(self, saveFilePath):
        torch.save(self.state_dict(), saveFilePath + ".pt")

    def load(self, saveFilePath):
        self.load_state_dict(torch.load(saveFilePath + ".pt"))

# <<abstract>>
    def forward(self, *arg):
        raise NotImplementedError()


class TestAgentTf(AgentTf):

    def __init__(self, Nout):
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, BatchNormalization

        super().__init__()
        self.filter = Sequential()
        self.filter.add(Dense(Nout))
        self.filter.add(BatchNormalization())

    def call(self, _X):
# _X: (*, Nin)
        _Y = self.filter(_X) # (*, 128)
        return _Y # (*, 128)


class TestAgentTorch(AgentTorch):

    def __init__(self, Nin, Nout):

        super().__init__()
        self.filter = nn.Sequential(
            nn.Linear(Nin, Nout))

    def forward(self, _X):
# _X: (*, Nin)
        _Y = self.filter(_X) # (*, 128)
        return _Y # (*, 128)
