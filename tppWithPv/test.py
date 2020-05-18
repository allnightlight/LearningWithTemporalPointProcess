
import unittest

import torch 

from history import History
import numpy as np
from tppWithPv.agentTpp import AgentHawkesWithPv
from tppWithPv.builderTpp import BuilderTpp
from tppWithPv.dataSet import TestPvDataSet, TestEventDataSet
from tppWithPv.environmentTpp import EventDataFeederWithPv
from tppWithPv.trainerTpp import TrainerTppMLE


class TestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        nSample = 2**10
        nDelta, nPv = np.random.randint(1, 2**2, size=(2,))
        dataRaw = np.random.randint(2, size=(nSample, nDelta))
        filePath = "./dataDelta.csv"
        with open(filePath, "w") as fp:
            fp.write("#\n")
            for row in dataRaw:                
                fp.write(",".join([str(cell) for cell in row]) + "\n")
                
        dataRaw = np.random.randn(nSample, nPv)
        filePath = "./dataPv.csv"
        with open(filePath, "w") as fp:
            fp.write("#\n")
            for row in dataRaw:                
                fp.write(",".join([str(cell) for cell in row]) + "\n")
                
        cls.nSample = nSample
        cls.nPv = nPv
        cls.nDelta = nDelta


    def test_001(self):

        for _ in range(2**7):
            Nseq, Nbatch, Nh, = np.random.randint(low=1
                , high=2**5
                , size=(3,))

            Ndelta = TestCase.nDelta
            Npv = TestCase.nPv

            agent = AgentHawkesWithPv(Ndelta, Nh, Npv)

            _E = torch.randn(Nseq, Nbatch, Ndelta)
            _Pv = torch.randn(Nseq, Nbatch, Npv)

            _I = agent(_E, _Pv)

            assert _I.shape == (Nseq+1, Nbatch, Ndelta)

            I = _I.data.numpy()

            assert not np.any(np.isnan(I))
            assert np.all((I >= 0) & (I <= 1))

    def test_002(self):

        pvDataSet = TestPvDataSet.getInstance()
        eventDataSet = TestEventDataSet.getInstance()
        for ds in [pvDataSet, eventDataSet]:
            N = ds.getNsample()
            idx = ds.getAvailableIndex()
            idx.shape == (N,)
            ds.getSlice(idx)

    def test_003(self):

        for _ in range(2**3):

            Nbatch, Nseq, = np.random.randint(low=1
                , high = 2**5
                , size=(2,)
                )

            environment = EventDataFeederWithPv(Nbatch , Nseq)

            for idx in environment.iterate():
                assert idx.shape == (Nbatch,)
                e, p = environment.getBatchData(idx)

                assert e.shape == (Nseq, Nbatch, TestCase.nDelta)
                assert p.shape == (Nseq, Nbatch, TestCase.nPv)

            idx = environment.getAvailableIndex()

    def test_004(self):

        for _ in range(2**4):
            Nseq, Nbatch, Nh,  = np.random.randint(low=1
                , high=2**3
                , size=(3,))

            environment = EventDataFeederWithPv(Nbatch , Nseq)
            
            agent = AgentHawkesWithPv(TestCase.nDelta, Nh, TestCase.nPv)

            Nepoch = 2
            trainer = TrainerTppMLE(agent, environment, Nepoch)
            trainer.train()

    def test_005(self):

        history = History("./test.db")
        builder = BuilderTpp(history)
        builder.build()

if __name__ == "__main__":
    unittest.main()
