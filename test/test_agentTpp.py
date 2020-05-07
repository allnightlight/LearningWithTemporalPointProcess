'''
Created on 2020/05/07

@author: ukai
'''

import sys
import unittest

import torch

from agentTpp import AgentHawkes, AgentFactoryTpp
import numpy as np


class Test(unittest.TestCase):

    def test_001(self):
        for k1 in range(2**7):
            sys.stdout.write("\r%04d" % k1)
            Ndelta, Nh, Nbatch  = np.random.randint(1, 2**5, size=(3,))
            Nseq = 2**7
            agent = AgentHawkes(Ndelta, Nh)
            _E = torch.rand((Nseq, Nbatch, Ndelta)) # (Nseq, *, Ndelta)
            _I = agent(_E) #(Nseq, *, Ndelta)
            I = _I.data.numpy()
            assert _I.shape == (Nseq+1, Nbatch, Ndelta)
            assert not np.any(np.isnan(I))
            assert np.all((I >= 0) & (I <= 1))

    def test_002(sself):
        Ndelta = 5
        Nh = 2**5
        agentType = "AgentHawkes"
        constructorParameter = dict(Ndelta = Ndelta, Nh = Nh)
        saveFilePath = "./tmp/test"

        agent = AgentHawkes(Ndelta, Nh)
        agent.save(saveFilePath)

        agent = AgentFactoryTpp().create(agentType, constructorParameter, \
            saveFilePath)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()