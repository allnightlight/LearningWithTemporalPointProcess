
from agent import TestAgent
from environment import TestEnvironment
from factory import AgentFactory, EnvironmentFactory, TrainerFactory
from history import IHistoryRequiredByLoader
from trainer import TestTrainer


# <<abstract>>
class Loader:

    def __init__(self, history, agentFactory, environmentFactory, 
        trainerFactory):
        assert isinstance(history, IHistoryRequiredByLoader)
        assert isinstance(agentFactory, AgentFactory)
        assert isinstance(environmentFactory, EnvironmentFactory)
        assert isinstance(trainerFactory, TrainerFactory)

        self.history = history
        self.agentFactory = agentFactory
        self.environmentFactory = environmentFactory
        self.trainerFactory = trainerFactory

# <<abstract>>
    def iterateTrainId(self):
        raise NotImplementedError
        yield 1
        yield 2

    def iterateHistory(self):
        it = self.iterateTrainId()
        for trainerId in it:

            agentId, environmentId, trainerType, trainConstructorParameter, \
                criteriaNames, trainLog = self.history.loadTrainer(trainerId)

            agentType, agentConstructorParameter, saveFilePath = \
                self.history.loadAgent(agentId)

            environmentType, environmentConstructorParameter = \
                self.history.loadEnvironment(environmentId)

            agent = self.agentFactory.create(agentType, \
                agentConstructorParameter, saveFilePath)

            environment = self.environmentFactory.create(environmentType, \
                environmentConstructorParameter)

            trainer = self.trainerFactory.create(trainerType, agent, \
                environment, trainConstructorParameter, criteriaNames, \
                trainLog)

            yield agent, environment, trainer




class TestHistory(IHistoryRequiredByLoader):

    def __init__(self):
        pass

    def loadAgent(self, agentId):
        return "TestAgent", {}, "SomewhereUsedToBe" 

    def loadEnvironment(self, environmentId):
        return "TestEnvironment", {}

    def loadTrainer(self, trainerId):
        return -1, -1, "TestTrainer", {}, None, None


class TestAgentFactory(AgentFactory):

    def initialize(self, classname, **args):
        return eval(classname)(**args)


class TestEnvironmentFactory(EnvironmentFactory):

    def initialize(self, classname, **args):
        return eval(classname)(**args)


class TestTrainerFactory(TrainerFactory):

    def initialize(self, classname, **args):
        return eval(classname)(**args)


class TestLoader(Loader):

    def __init__(self, history, agentFactory, environmentFactory, \
        trainerFactory):
        super().__init__(history, agentFactory, environmentFactory, \
        trainerFactory)

    def iterateTrainId(self):
        yield 1
        yield 2
        yield 3
