
import sys

import torch

from agent import AgentTorch
from factory import AgentFactory
import numpy as np
import torch.nn as nn


class AgentFactoryTpp(AgentFactory):

    def initialize(self, classname, **args):
        return eval(classname)(**args)


class AgentHawkes(AgentTorch):

    def __init__(self, Ndelta, Nh):
        super().__init__()
# _A: (Nh,)
# _B: (Ndelta, Nh,)
# _C: (Nh, Ndelta,)
        self.Nh = Nh

        #self._loglogT = nn.Parameter(torch.randn(Nh, requires_grad = True)) # (Nh,)
        self._loglogT = nn.Parameter(torch.randn(Nh)) # (Nh,)
        self._B = nn.Linear(Ndelta, Nh) # (Ndelta -> Nh)
        self._C = nn.Linear(Nh, Ndelta) # (Nh -> Ndelta)

    def forward(self, _E):
# _E: (Nseq, *, Ndelta)
# _I: (Nseq, *, Ndelta) Intensity

        _invT = torch.exp(-torch.exp(self._loglogT)) # (Nh,), 1 - 1/T

        Nseq, Nbatch, _ = _E.shape
        I = []

        _H = torch.zeros(Nbatch, self.Nh) # (*, Nh)
        I.append(torch.sigmoid(self._C(_H))) # (*, Ndelta)
        for k1 in range(Nseq):
            _H  = _H -  _H * _invT + self._B(_E[k1,:,:]) # (*, Nh)
            I.append(torch.sigmoid(self._C(_H))) # (*, Ndelta)

        _I = torch.stack(I, dim=0) # (Nseq+1, *, Ndelta)

        return _I # (Nseq+1, *, Ndelta)
