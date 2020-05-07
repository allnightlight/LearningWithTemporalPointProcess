'''
Created on 2020/05/07

@author: ukai
'''
from datetime import datetime
import unittest

from agentTpp import AgentHawkes
from environmentTpp import EventDataFeeder, SingleEventDataFeeder
from trainerTpp import TrainerTpp, TrainerTppSingleDelta, TrainerTppMLE


class Test(unittest.TestCase):

    def test_001(self):

        Nh      = 2**0
        Nbatch  = 2**6
        Nseq    = 2**4
        Nepoch  = 2**1

        environment = EventDataFeeder(Nbatch, Nseq) 
        Ndelta = environment.Ndelta

        agent = AgentHawkes(Ndelta, Nh)
        trainer = TrainerTpp(agent, environment, Nepoch)

        trainer.train()
        criteriaNames = trainer.getCriteriaNames()
        timestamp, tbl = trainer.getTrainLog()

        assert isinstance(criteriaNames[0], str)
        assert isinstance(timestamp[0], datetime)
        assert len(criteriaNames) == len(tbl[0])
        assert len(timestamp) == len(tbl)

    def test_002(self):

        Nh      = 2**0
        Nbatch  = 2**6
        Nseq    = 2**4
        Nepoch  = 2**1
        Ndelta  = 1

        environment = SingleEventDataFeeder(Nbatch, Nseq) 

        agent = AgentHawkes(Ndelta, Nh)
        trainer = TrainerTppSingleDelta(agent, environment, Nepoch)

        trainer.train()
        criteriaNames = trainer.getCriteriaNames()
        timestamp, tbl = trainer.getTrainLog()

        assert isinstance(criteriaNames[0], str)
        assert isinstance(timestamp[0], datetime)
        assert len(criteriaNames) == len(tbl[0])
        assert len(timestamp) == len(tbl)

    def test_003(self):

        Nh      = 2**0
        Nbatch  = 2**6
        Nseq    = 2**4
        Nepoch  = 2**1

        environment = EventDataFeeder(Nbatch, Nseq) 
        Ndelta = environment.Ndelta

        agent = AgentHawkes(Ndelta, Nh)
        trainer = TrainerTppMLE(agent, environment, Nepoch)

        trainer.train()
        criteriaNames = trainer.getCriteriaNames()
        timestamp, tbl = trainer.getTrainLog()

        assert isinstance(criteriaNames[0], str)
        assert isinstance(timestamp[0], datetime)
        assert len(criteriaNames) == len(tbl[0])
        assert len(timestamp) == len(tbl)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()