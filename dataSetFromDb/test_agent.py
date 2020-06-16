'''
Created on 2020/06/15

@author: ukai
'''
import unittest
from agentGru import AgentGru
import torch 
import numpy as np
import itertools

class Test(unittest.TestCase):


    def test001(self):
        
        for (Npv, Nseq, Ndelta, Nh)
            in itertools.product(
                (10, 0)
                , (1, 2**3)
                , (1, 3)
                , (2**5,)
                )
        
            args = dict(
                Ndelta = Ndelta
                , Npv = Npv
                , Nh = Nh)
            
            agent = AgentGru(**args)
            
            _E = torch.rand(Nseq, Nbatch, Ndelta) # (Nseq, *, Ndelta)         
            _Pv = torch.randn(Nseq, Nbatch, Npv) # (Nseq, *, Npv)
            
            _Phat, _Yhat = agent(_E, _Pv)
            
            assert _Phat.shape == (Nseq+1, Nbatch, Ndelta)
            assert _Yhat.shape == (Nseq+1, Nbatch, Ndelta)
            
            Phat = _Phat.data.numpy()
            assert np.all((Phat >= -1e-16) & (Phat <= 1+1e-16))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()