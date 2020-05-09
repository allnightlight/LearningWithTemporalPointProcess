
import sys
sys.path.append('../framework')

import torch

from agent import AgentTorch
from factory import AgentFactory
import numpy as np
import torch.nn as nn


class AgentFactoryTpp(AgentFactory):

    def initialize(self, classname, **args):
        return eval(classname)(**args)


class AgentHawkesWithPv(AgentTorch):

    def __init__(self, Ndelta, Nh, Npv):
        super().__init__()
# _A: (Nh,): hidden vec. -> hidden vec.
# _B: (Ndelta + Npv, Nh,): (delta, pv) -> hidden vec.
# _C: (Nh, Ndelta,): hidden vec. -> delta
        self.Nh = Nh

        self._loglogT = nn.Parameter(torch.randn(Nh)) # (Nh,)
        self._B = nn.Linear(Ndelta + Npv, Nh) # (Ndelta + Npv -> Nh)
        self._C = nn.Linear(Nh, Ndelta) # (Nh -> Ndelta)

    def forward(self, _E, _Pv):
# _E: (Nseq, *, Ndelta)
# _Pv: (Nseq, *, Npv)
# _I: (Nseq, *, Ndelta) Intensity

        _oneOverT = torch.exp(-torch.exp(self._loglogT)) # (Nh,),  = 1/T

        Nseq, Nbatch, _ = _E.shape
        I = []

        _H = torch.zeros(Nbatch, self.Nh) # (*, Nh)
        I.append(torch.sigmoid(self._C(_H))) # (*, Ndelta)
        for k1 in range(Nseq):
            _H  = _H -  _H * _oneOverT + self._B(
                torch.cat((_E[k1,:,:]
                    , _Pv[k1,:,:]) 
                    , dim = -1)
                ) # (*, Nh)
            I.append(torch.sigmoid(self._C(_H))) # (*, Ndelta)

        _I = torch.stack(I, dim=0) # (Nseq+1, *, Ndelta)

        return _I # (Nseq+1, *, Ndelta)
