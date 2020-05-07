
from adapter import IAgentAdapter, IEnvironmentAdapter, ITrainerAdapter
from history import IHistory


# <<abstract>>
class Builder:

    def __init__(self, history):
        assert isinstance(history, IHistory)

        self.history = history
        self.saveFilePathFormat = "./tmp/agent_id={0:04d}"

    def iterate(self):
        raise NotImplementedError

    def build(self):
        it = self.iterate()
        rtn = []
        for agentAdapter, environmentAdapter, trainerAdapter in it:
            assert isinstance(agentAdapter, IAgentAdapter)
            assert isinstance(environmentAdapter, IEnvironmentAdapter)
            assert isinstance(trainerAdapter, ITrainerAdapter)

            trainerAdapter.train()

# save the agent:
            agentId = self.history.getNewAgentId()
            saveFilePath = self.saveFilePathFormat.format(agentId)
            agentAdapter.save(saveFilePath)

            self.history.saveAgent(agentId, 
                agentAdapter.getType(),
                agentAdapter.getConstructorParameter(),
                saveFilePath)

# save the environment:
            environmentId = self.history.getNewEnvironmentId()
            self.history.saveEnvironment(environmentId, 
                environmentAdapter.getType(),
                environmentAdapter.getConstructorParameter(),
                )

# save the trainer:
            trainerId = self.history.getNewTrainerId()
            self.history.saveTrainer(trainerId, 
                agentId,
                environmentId,
                trainerAdapter.getType(),
                trainerAdapter.getConstructorParameter(),
                trainerAdapter.getCriteriaNames(),
                trainerAdapter.getTrainLog(),
                )

            rtn.append((agentId, environmentId, trainerId))
        return rtn


# <<concrete>>
class TestBuilder(Builder):

    def __init__(self, history):
        super().__init__(history)

    def iterate(self):
        from adapter import TestAgentAdapter, TestEnvironmentAdapter, \
            TestTrainerAdapter
        for k1 in range(10):
            yield TestAgentAdapter(), TestEnvironmentAdapter(), \
                TestTrainerAdapter()

