'''
Created on 2020/06/18

@author: ukai
'''
import os
import unittest

from agentGru import AgentGru, AgentFactoryGru
from environmentFromDb import EnvironmentFactoryFromDb
from evaluator import Evaluator
from history import History
import numpy as np
from trainerMLE import TrainerMLEFactory


class Test(unittest.TestCase):

    def test001(self):
        
        nBatch = 2**7
        nDelta = 2**0
        
        Eref = np.random.randint(2, size=(nBatch, nDelta))
        Eest = np.random.randint(2, size=(nBatch, nDelta))
        
        for nTol in range(3):
            
            count, prop = Evaluator.countFpAndFn(Eref, Eest, nTol)
            
            assert count.shape == prop.shape
            assert count.shape == (4, nDelta)

    def test002(self):

        nPv = 8
        nDelta = 2
        nBatch = 2**7
        nSeq = 8

        for nTol in range(3):
                    
            agent = AgentGru(Ndelta = nDelta, Npv = nPv, Nh = 2**5)
            
            Eref = np.random.randint(2, size=(nSeq, nBatch, nDelta)).astype(np.float32)
            Pv = np.random.randn(nSeq, nBatch, nPv).astype(np.float32)
    
            count, prop = Evaluator.evaluateAnAgent(agent, Eref, Pv, nTol)
    
            assert count.shape == prop.shape
            assert count.shape == (4, nDelta)        
        
    def test003(self):
        
        
        history = History("./tmp/test.db")
        
        agentFactory = AgentFactoryGru()
        environmentFactory = EnvironmentFactoryFromDb()
        trainerFactory = TrainerMLEFactory()

        for nTol in range(3):        
            evaluator = Evaluator(history, agentFactory, environmentFactory, trainerFactory, nTol)
            
            evaluator.evaluate("test.csv")
            
            os.remove('test.csv')        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()