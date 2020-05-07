
from agent import IAgent
from environment import IEnvironment
from trainer import ITrainer


# <<abstract>>
class AgentFactory():

    def create(self, agentType, constructorParameter, saveFilePath):
        agent = self.initialize(agentType, **constructorParameter)
        assert isinstance(agent, IAgent)
        if saveFilePath is not None:
            agent.load(saveFilePath)
        return agent

# <<abstract>>
    def initialize(self, classname, **args):
        raise NotImplementedError
        return eval(classname)(**args)


# <<abstract>>
class EnvironmentFactory():

    def create(self, environmentType, constructorParameter):
        environment = self.initialize(environmentType, **constructorParameter)
        assert isinstance(environment, IEnvironment)
        return environment

# <<abstract>>
    def initialize(self, classname, **args):
        raise NotImplementedError
        return eval(classname)(**args)


# <<abstract>>
class TrainerFactory():

    def create(self, trainerType, agent, environment, constructorParameter, \
        criteriaNames, trainLog):
        assert isinstance(agent, IAgent)
        assert isinstance(environment, IEnvironment)
        trainer = self.initialize(trainerType, agent = agent,\
            environment = environment, **constructorParameter)
        assert isinstance(trainer, ITrainer)
        trainer.setCriteriaNames(criteriaNames)
        trainer.setTrainLog(trainLog)
        return trainer

# <<abstract>>
    def initialize(self, classname, **args):
        raise NotImplementedError
        return eval(classname)(**args)


