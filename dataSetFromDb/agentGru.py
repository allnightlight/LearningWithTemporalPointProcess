
import sys
sys.path.append('../framework')

import torch

from agent import AgentTorch
from factory import AgentFactory
import numpy as np
import torch.nn as nn


class AgentFactoryGru(AgentFactory):

    def initialize(self, classname, **args):
        return eval(classname)(**args)


class AgentGru(AgentTorch):

    def __init__(self, Ndelta, Npv, Nh):
        super().__init__()
        self.Nh = Nh
        
        self.gru = nn.GRU(Ndelta + Npv, Nh)
        self.fnn = nn.Sequential(nn.Linear(Nh, Nh)
                                 , nn.Tanh()
                                 , nn.Linear(Nh, Ndelta))


    def forward(self, _E, _Pv):
# _E: (Nseq, *, Ndelta)
# _Pv: (Nseq, *, Npv)

        Nh = self.Nh
        Nbatch = _E.shape[1]

        _X = torch.cat((_E, _Pv), axis=-1) # (Nseq, *, Ndelta + Npv)
        _H0 = torch.zeros(1, Nbatch, Nh) # (1, *, Nh)
        
        _H1, _ = self.gru(_X, _H0) # _H1: (Nseq, *, Nh)
        _H = torch.cat((_H0, _H1), axis=0) # (Nseq+1, *, Nh)
        
        _Yhat = self.fnn(_H) # (Nseq+1, *, Ndelta), logit
        _Phat = torch.sigmoid(_Yhat) # (Nseq+1, *, Ndelta), prob.

        return _Phat, _Yhat # (Nseq+1, *, Ndelta)
