
from datetime import datetime

from agent import IAgent
from environment import IEnvironment
from trainer import ITrainer


# <<interface>>
class IAgentAdapter:

    def getType(self):
        raise NotImplementedError
        return "String"

    def getConstructorParameter(self):
        raise NotImplementedError
        return dict(name = "int/float/datetime",)

    def save(self, saveFilePath):
        raise NotImplementedError

    def load(self, saveFilePath):
        raise NotImplementedError

# <<interface>>
class IEnvironmentAdapter:

    def getType(self):
        raise NotImplementedError
        return "String"

    def getConstructorParameter(self):
        raise NotImplementedError
        return dict(name = "int/float/datetime",)

# <<interface>>
class ITrainerAdapter:

    def getType(self):
        raise NotImplementedError
        return "String"

    def getConstructorParameter(self):
        raise NotImplementedError
        return dict(name = "int/float/datetime",)

    def train(self):
        raise NotImplementedError

    def getCriteriaNames(self):
        raise NotImplementedError
        return ("RMSE", "MAPE",)

    def getTrainLog(self):
        raise NotImplementedError
        timestamp = (datetime.now(), )
        tbl = ((1.0, 2.0,),)
        return timestamp, tbl


class TestAgentAdapter(IAgentAdapter):
    def __init__(self):
        super().__init__()

    def getType(self):
        return "TestAgent"

    def getConstructorParameter(self):
        return dict(param1 = 123, param2 = 1.23, param3 = datetime.now())

    def save(self, saveFilePath):
        pass

    def load(self, saveFilePath):
        pass

class TestEnvironmentAdapter(IEnvironmentAdapter):

    def getType(self):
        return "TestEnvironment"

    def getConstructorParameter(self):
        return dict(name = "int/float/datetime",)


class TestTrainerAdapter(ITrainerAdapter):

    def getType(self):
        return "TestTrainer"

    def getConstructorParameter(self):
        return dict(name = "int/float/datetime",)

    def train(self):
        pass

    def getCriteriaNames(self):
        return ("RMSE", "MAPE",)

    def getTrainLog(self):
        timestamp = (datetime.now(), )
        tbl = ((1.0, 2.0,),)
        return timestamp, tbl


class AgentAdapter(IAgentAdapter):
    
    def __init__(self, agent):
        super().__init__()
        assert isinstance(agent, IAgent)
        self.agent = agent
        self.constructorParameter = None

    def getType(self):
        return self.agent.__class__.__name__

    def getConstructorParameter(self):
        assert self.constructorParameter is not None
        return self.constructorParameter

    def setConstructorParameter(self, constructorParameter):
        self.constructorParameter = constructorParameter

    def save(self, saveFilePath):
        self.agent.save(saveFilePath)

    def load(self, saveFilePath):
        self.agent.load(saveFilePath)


class EnvironmentAdapter(IEnvironmentAdapter):

    def __init__(self, environment):
        super().__init__()
        assert isinstance(environment, IEnvironment)
        self.environment = environment
        self.constructorParameter = None

    def getType(self):
        return self.environment.__class__.__name__

    def getConstructorParameter(self):
        assert self.constructorParameter is not None
        return self.constructorParameter

    def setConstructorParameter(self, constructorParameter):
        self.constructorParameter = constructorParameter


class TrainerAdapter(ITrainerAdapter):

    def __init__(self, trainer):
        super().__init__()
        assert isinstance(trainer, ITrainer)
        self.trainer = trainer 
        self.constructorParameter = None

    def getType(self):
        return self.trainer.__class__.__name__

    def getConstructorParameter(self):
        assert self.constructorParameter is not None
        return self.constructorParameter

    def setConstructorParameter(self, constructorParameter):
        self.constructorParameter = constructorParameter

    def train(self):
        self.trainer.train()

    def getCriteriaNames(self):
        return self.trainer.getCriteriaNames()

    def getTrainLog(self):
        return self.trainer.getTrainLog()