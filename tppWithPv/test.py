
import sys
sys.path.append('../framework')

import unittest

import numpy as np
import torch 
from agentTpp import AgentHawkesWithPv
from dataSet import TestEventDataSet, TestPvDataSet
from environmentTpp import EventDataFeederWithPv
from trainerTpp import TrainerTppMLE
from builderTpp import BuilderTpp

from history import History

class TestCase(unittest.TestCase):

    def test_001(self):

        for _ in range(2**7):
            Nseq, Nbatch, Ndelta, Nh, Npv = np.random.randint(low=1
                , high=2**5
                , size=(5,))

            agent = AgentHawkesWithPv(Ndelta, Nh, Npv)

            _E = torch.randn(Nseq, Nbatch, Ndelta)
            _Pv = torch.randn(Nseq, Nbatch, Npv)

            _I = agent(_E, _Pv)

            assert _I.shape == (Nseq+1, Nbatch, Ndelta)

            I = _I.data.numpy()

            assert not np.any(np.isnan(I))
            assert np.all((I >= 0) & (I <= 1))

    def test_002(self):

        Ndelta = 5
        Npv = 3
        N = 2**10

        pvDataSet = TestPvDataSet(Ndelta)
        eventDataSet = TestEventDataSet(Npv)
        for ds in [pvDataSet, eventDataSet]:
            assert ds.getNsample() == N
            idx = ds.getAvailableIndex()
            idx.shape == (N,)
            ds.getSlice(idx)

    def test_003(self):

        for _ in range(2**3):

            Nbatch, Nseq, Ndelta, Npv = np.random.randint(low=1
                , high = 2**5
                , size=(4,)
                )

            environment = EventDataFeederWithPv(Ndelta,Npv, Nbatch , Nseq)

            for idx in environment.iterate():
                assert idx.shape == (Nbatch,)
                e, p = environment.getBatchData(idx)

                assert e.shape == (Nseq, Nbatch, Ndelta)
                assert p.shape == (Nseq, Nbatch, Npv)

            idx = environment.getAvailableIndex()

    def test_004(self):

        for _ in range(2**4):
            Nseq, Nbatch, Ndelta, Nh, Npv = np.random.randint(low=1
                , high=2**3
                , size=(5,))

            environment = EventDataFeederWithPv(Ndelta, Npv, Nbatch , Nseq)

            agent = AgentHawkesWithPv(Ndelta, Nh, Npv)

            Nepoch = 2
            trainer = TrainerTppMLE(agent, environment, Nepoch)
            trainer.train()

    def test_005(self):

        history = History("./test.db")
        builder = BuilderTpp(history)
        builder.build()


if __name__ == "__main__":
    unittest.main()
