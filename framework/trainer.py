
from datetime import datetime

from agent import IAgent
from environment import IEnvironment


# <<interface>>
class ITrainer:

    def train(self):
        raise NotImplementedError

    def getCriteriaNames(self):
        raise NotImplementedError
        return ("LogLikelihood", "FalseNeg", "FalsePos", )

    def getTrainLog(self):
        raise NotImplementedError
        timestamp = (datetime.now(), )
        tbl = ((1.0, 0.9, 0.1),)
        return timestamp, tbl

    def setCriteriaNames(self, criteriaNames):
        raise NotImplementedError

    def setTrainLog(self, trainLog):
        raise NotImplementedError


class TestTrainer(ITrainer):

    def __init__(self, agent, environment):
        super().__init__()
        assert isinstance(agent, IAgent)
        assert isinstance(environment, IEnvironment)
        self.agent = agent
        self.environment = environment

    def train(self):
        pass

    def getCriteriaNames(self):
        return ["LogLikelihood", "FalseNeg", "FalsePos", ]

    def getTrainLog(self):
        timestamp = [datetime.now(), ]
        tbl = [[1.0, 0.9, 0.1],]
        return timestamp, tbl

    def setCriteriaNames(self, criteriaNames):
        pass

    def setTrainLog(self, trainLog):
        pass