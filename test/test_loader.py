'''
Created on 2020/05/07

@author: ukai
'''
import unittest

from agent import TestAgent
from environment import TestEnvironment
from loader import TestAgentFactory, TestEnvironmentFactory, TestTrainerFactory, \
    TestLoader, TestHistory
from trainer import TestTrainer


class Test(unittest.TestCase):


    def test_001(self):

        history = TestHistory()
        agentFactory = TestAgentFactory()
        environmentFactory = TestEnvironmentFactory()
        trainerFactory = TestTrainerFactory()
        loader = TestLoader(history, agentFactory, environmentFactory, \
            trainerFactory)

        for agent, environment, trainer in loader.iterateHistory():
            assert isinstance(agent, TestAgent)
            assert isinstance(environment, TestEnvironment)
            assert isinstance(trainer, TestTrainer)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()