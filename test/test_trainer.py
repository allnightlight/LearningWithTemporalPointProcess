'''
Created on 2020/05/07

@author: ukai
'''
from datetime import datetime
import unittest

from agent import TestAgent
from environment import TestEnvironment
from trainer import TestTrainer


class Test(unittest.TestCase):

    def test_001(self):

        agent = TestAgent()
        environment = TestEnvironment()

        trainer = TestTrainer(agent, environment)

        trainer.train()
        criteriaNames = trainer.getCriteriaNames()
        assert isinstance(criteriaNames, list)
        assert isinstance(criteriaNames[0], str)

        trainLog = trainer.getTrainLog()
        assert isinstance(trainLog[0][0], datetime)
        assert isinstance(trainLog[1][0][0], float)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()