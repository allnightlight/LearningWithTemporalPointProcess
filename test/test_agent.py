'''
Created on 2020/05/07

@author: ukai
'''

import unittest

import torch 

from agent import TestAgent, TestAgentTf, TestAgentTorch
import tensorflow as tf


class Test(unittest.TestCase):


    def test_001(self):
        agent = TestAgent()

    def test_002(self):
        import numpy as np

        Nout = 128
        Nin = 10
        Nbatch = 2**5

        _X = tf.random.normal(shape = (Nbatch, Nin,)) # (*, Nin)

        constructorParameter = dict(Nout = Nout)

        agent1 = TestAgentTf(**constructorParameter)
        agent2 = TestAgentTf(**constructorParameter)
        agent3 = TestAgentTf(**constructorParameter)

        _Y1 = agent1(_X) # (*, Nout)
        Y1 = _Y1.numpy() # (*, Nout)
        _Y2 = agent2(_X) # (*, Nout)
        Y2 = _Y2.numpy() # (*, Nout)

        agent1.save("./tmp/agent1")
        agent2.save("./tmp/agent2")

        agent3.load("./tmp/agent1") # should be same with agent1

        _Y3 = agent3(_X) # (*, Nout)
        Y3 = _Y3.numpy() # (*, Nout)

        assert np.all(Y3 == Y1)

    def test_003(self):
        import numpy as np

        Nout = 128
        Nin = 10
        Nbatch = 2**5

        _X = torch.randn(size = (Nbatch, Nin,)) # (*, Nin)

        constructorParameter = dict(Nin = Nin, Nout = Nout)

        agent1 = TestAgentTorch(**constructorParameter)
        agent2 = TestAgentTorch(**constructorParameter)
        agent3 = TestAgentTorch(**constructorParameter)

        _Y1 = agent1(_X) # (*, Nout)
        Y1 = _Y1.detach().numpy() # (*, Nout)
        _Y2 = agent2(_X) # (*, Nout)
        Y2 = _Y2.detach().numpy() # (*, Nout)

        agent1.save("./tmp/agent1")
        agent2.save("./tmp/agent2")

        agent3.load("./tmp/agent1") # should be same with agent1

        _Y3 = agent3(_X) # (*, Nout)
        Y3 = _Y3.detach().numpy() # (*, Nout)

        assert np.all(Y3 == Y1)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()