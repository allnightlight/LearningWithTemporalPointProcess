
import os
import unittest

import torch 

from history import History
import numpy as np
from tppWithPv.agentTpp import AgentHawkesWithPv
from tppWithPv.builderTpp import BuilderTpp
from tppWithPv.dataSet import TestPvDataSet, TestEventDataSet
from tppWithPv.environmentTpp import EventDataFeederWithPv
from tppWithPv.trainerTpp import TrainerTppMLE
import shutil


class TestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        if not os.path.exists('./tmp'):
            os.mkdir("./tmp")
        
        nSample = 2**6
        nDelta, nPv = np.random.randint(1, 2**2, size=(2,))
        dataRaw = np.random.randint(2, size=(nSample, nDelta))
        eventDataFilePath = "./tmp/dataDelta.csv"
        with open(eventDataFilePath, "w") as fp:
            fp.write("#\n")
            for row in dataRaw:                
                fp.write(",".join([str(cell) for cell in row]) + "\n")
                
        dataRaw = np.random.randn(nSample, nPv)
        pvDataFilePath = "./tmp/dataPv.csv"
        with open(pvDataFilePath, "w") as fp:
            fp.write("#\n")
            for row in dataRaw:                
                fp.write(",".join([str(cell) for cell in row]) + "\n")
                
        cls.nSample = nSample
        cls.nPv = nPv
        cls.nDelta = nDelta
        cls.eventDataFilePath = eventDataFilePath
        cls.pvDataFilePath = pvDataFilePath
        
    @classmethod
    def tearDownClass(cls):
        if os.path.exists('./tmp'):
            shutil.rmtree("./tmp")



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

        pvDataSet = TestPvDataSet.getInstance(TestCase.pvDataFilePath)
        eventDataSet = TestEventDataSet.getInstance(TestCase.eventDataFilePath)
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

            environment = EventDataFeederWithPv(TestCase.eventDataFilePath, TestCase.pvDataFilePath, Nbatch , Nseq)

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

            environment = EventDataFeederWithPv(TestCase.eventDataFilePath, TestCase.pvDataFilePath, Nbatch , Nseq)
            
            agent = AgentHawkesWithPv(TestCase.nDelta, Nh, TestCase.nPv)

            Nepoch = 2
            trainer = TrainerTppMLE(agent, environment, Nepoch)
            trainer.train()

    def test_005(self):

        for _ in range(2**4):
            Nseq, Nbatch, Nh,  = np.random.randint(low=1
                , high=2**3
                , size=(3,))

            environment = EventDataFeederWithPv(TestCase.eventDataFilePath, TestCase.pvDataFilePath, Nbatch , Nseq, preprocess="Differential", tau = 10)
            
            agent = AgentHawkesWithPv(TestCase.nDelta, Nh, TestCase.nPv)

            Nepoch = 2
            trainer = TrainerTppMLE(agent, environment, Nepoch)
            trainer.train()


if __name__ == "__main__":
    unittest.main()
