
from datetime import datetime
import sys

import torch

from factory import TrainerFactory
import numpy as np
from trainer import ITrainer


class TrainerMLEFactory(TrainerFactory):

    def initialize(self, classname, **args):
        return eval(classname)(**args)


class TrainerMLE(ITrainer):

    def __init__(self, agent, environment, Nepoch):
        super().__init__()
        self.agent = agent
        self.Nepoch = Nepoch
        self.environment = environment
        self.optimizer = torch.optim.Adam(agent.parameters())

        self.criteriaNames = ["Epoch", "LogLikelihood",]
        self.timestamp = []
        self.tbl = []

    def train(self):
        sys.stdout.write("\nA train session started.\n")
        for epoch in range(self.Nepoch):
            lossHistory = []
            t_bgn = datetime.now()
            sys.stdout.write("\r%s Epoch %03d, " % (str(t_bgn), epoch))
            for idx in self.environment.iterate():
                E, Pv = self.environment.getBatchData(idx) # (Nseq, *, Ndelta)
                _E = torch.tensor(E.astype(np.float32)) # (Nseq, *, Ndelta)
                _Pv = torch.tensor(Pv.astype(np.float32)) # (Nseq, *, nPv)
                _, _Yhat = self.agent(_E, _Pv) # (Nseq+1, *, Ndelta)
                
                _box1 = -1. * torch.sum((1. - _E) * _Yhat[:-1,:], axis=-1) # (Nseq, *)
                _box2 = torch.sum(torch.nn.functional.logsigmoid(_Yhat[:-1,:]), axis=-1) # (Nseq, *)
                _LL = torch.mean(_box1 + _box2) # (,)
                
                _loss = -1. * _LL
                lossHistory.append(float(_loss))

                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()

            lossAvg = np.mean(lossHistory)
            t_end = datetime.now()
            sys.stdout.write("Loss %8.2f, ElapsedTime %s" 
                % (lossAvg, str(t_end-t_bgn)))

            self.timestamp.append(datetime.now())
            self.tbl.append((epoch, lossAvg,))
        sys.stdout.write("\nThe train session ended.\n")

    def getCriteriaNames(self):
        return self.criteriaNames

    def getTrainLog(self):
        return self.timestamp, self.tbl

    def setCriteriaNames(self, criteriaNames):
        if criteriaNames is not None:
            self.criteriaNames = criteriaNames

    def setTrainLog(self, trainLog):
        if trainLog is not None:
            self.timestamp, self.tbl = trainLog
