
from datetime import datetime
import sys

import torch

from agentTpp import AgentHawkes
from environmentTpp import EventDataFeeder, SingleEventDataFeeder
from factory import TrainerFactory
import numpy as np
from trainer import ITrainer


class TrainerFactoryTpp(TrainerFactory):

    def initialize(self, classname, **args):
        return eval(classname)(**args)


class TrainerTpp(ITrainer):

    def __init__(self, agent, environment, Nepoch):
        super().__init__()
        assert isinstance(agent, AgentHawkes)
        assert isinstance(environment, EventDataFeeder)
        self.agent = agent
        self.Nepoch = Nepoch
        self.environment = environment
        self.optimizer = torch.optim.Adam(agent.parameters())

        self.criteriaNames = ["Epoch", "CountingMeasure",]
        self.timestamp = []
        self.tbl = []

    def train(self):
        sys.stdout.write("\nA train session started.\n")
        for epoch in range(self.Nepoch):
            lossHistory = []
            t_bgn = datetime.now()
            sys.stdout.write("\r%s Epoch %03d, " % (str(t_bgn), epoch))
            for idx in self.environment.iterate():
                E = self.environment.getBatchData(idx) # (Nseq, *, Ndelta)
                _E = torch.tensor(E.astype(np.float32)) # (Nseq, *, Ndelta)
                _I = self.agent(torch.tensor(E)) # (Nseq+1, *, Ndelta)

                _C = torch.cumsum(_E, dim=0) # (Nseq, *, Ndelta)
                _Chat = torch.cumsum(_I[:-1,:,:], dim=0) # (Nseq, *, Ndelta)

                _loss = torch.mean(torch.abs(_C-_Chat)) 
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
        self.criteriaNames = criteriaNames

    def setTrainLog(self, trainLog):
        self.timestamp, self.tbl = trainLog


class TrainerTppSingleDelta(ITrainer):

    def __init__(self, agent, environment, Nepoch):
        super().__init__()
        assert isinstance(agent, AgentHawkes)
        assert isinstance(environment, SingleEventDataFeeder)
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
                E = self.environment.getBatchData(idx) # (Nseq, *, Ndelta)
                _E = torch.tensor(E.astype(np.float32)) # (Nseq, *, Ndelta)
                _I = self.agent(torch.tensor(E)) # (Nseq+1, *, Ndelta)

                _LL = torch.mean(torch.log(_I[1:-1,:,:]) * _E[1:,:,:] + \
                    torch.log(1.-_I[1:-1,:,:]) * (1-_E[1:,:,:]))

                _loss = -_LL
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


class TrainerTppMLE(ITrainer):

    def __init__(self, agent, environment, Nepoch):
        super().__init__()
        assert isinstance(agent, AgentHawkes)
        assert isinstance(environment, EventDataFeeder)
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
                E = self.environment.getBatchData(idx) # (Nseq, *, Ndelta)
                _E = torch.tensor(E.astype(np.float32)) # (Nseq, *, Ndelta)
                _I = self.agent(torch.tensor(E)) # (Nseq+1, *, Ndelta)

                _LL = torch.mean(torch.log(_I[:-1,:,:]) * _E + \
                    torch.log(1.-_I[:-1,:,:]) * (1-_E))

                _loss = -_LL
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
