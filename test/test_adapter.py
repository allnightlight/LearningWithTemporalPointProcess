'''
Created on 2020/05/07

@author: ukai
'''
from datetime import datetime
import unittest

from adapter import TestAgentAdapter, TestEnvironmentAdapter, TestTrainerAdapter, \
    AgentAdapter, EnvironmentAdapter, TrainerAdapter


class Test(unittest.TestCase):

    def test_001(self):
        agentAdapter = TestAgentAdapter()

    def test_002(self):
        environmentAdapter = TestEnvironmentAdapter()

    def test_003(self):
        trainerAdapter = TestTrainerAdapter()

    def test_004(self):
        from agent import TestAgent
        agent = TestAgent()
        agentAdapter = AgentAdapter(agent)

        constructorParameter = dict(N = 10, eps = 0.1, t = datetime.now())
        agentAdapter.setConstructorParameter(constructorParameter)
        assert constructorParameter == agentAdapter.getConstructorParameter()
        assert agentAdapter.getType() == agent.__class__.__name__

        agentAdapter.save("file name")
        agentAdapter.load("file name")

    def test_005(self):
        from environment import TestEnvironment
        environment = TestEnvironment()
        environmentAdapter = EnvironmentAdapter(environment)

        constructorParameter = dict(N = 10, eps = 0.1, t = datetime.now())
        environmentAdapter.setConstructorParameter(constructorParameter)
        assert constructorParameter == \
            environmentAdapter.getConstructorParameter()
        assert environmentAdapter.getType() == environment.__class__.__name__

    def test_006(self):
        from agent import TestAgent
        from environment import TestEnvironment
        from trainer import TestTrainer
        agent = TestAgent()
        environment = TestEnvironment()
        trainer = TestTrainer(agent, environment)
        trainerAdapter = TrainerAdapter(trainer)

        constructorParameter = dict(N = 10, eps = 0.1, t = datetime.now())
        trainerAdapter.setConstructorParameter(constructorParameter)
        assert constructorParameter == \
            trainerAdapter.getConstructorParameter()
        assert trainerAdapter.getType() == trainer.__class__.__name__

        trainerAdapter.getCriteriaNames()
        trainerAdapter.getTrainLog()



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()