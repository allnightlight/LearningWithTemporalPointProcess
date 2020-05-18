

from environment import IEnvironment
from factory import EnvironmentFactory
import numpy as np
from tppWithPv.dataSet import TestPvDataSet, TestEventDataSet


class EnvironmentFactoryTpp(EnvironmentFactory):

    def initialize(self, classname, **args):
        return eval(classname)(**args)


class EventDataFeederWithPv(IEnvironment):

    def __init__(self, Nbatch, Nseq):
        super().__init__()

        pvDataSet = TestPvDataSet.getInstance()
        eventDataSet = TestEventDataSet.getInstance()

        self.eventDataSet = eventDataSet
        self.pvDataSet    = pvDataSet
        self.Nbatch = Nbatch
        self.Nseq = Nseq

        assert min(self.eventDataSet.getNsample()
            , self.pvDataSet.getNsample())  > Nseq

    def getAvailableIndex(self):
        i0 = self.eventDataSet.getAvailableIndex()
        i1 = self.pvDataSet.getAvailableIndex()
        idxAvailable = np.intersect1d(i0, i1)
        idxAvailable = [ i for i in idxAvailable
            if np.all(np.isin(np.arange(i+1-self.Nseq, i+1), idxAvailable)) ]
        return idxAvailable # (almost N-Nseq+1,)

    def iterate(self):
        idxAvailable = self.getAvailableIndex() # (almost N-Nseq+1,)
        Navailable = len(idxAvailable)
        for _ in range(Navailable//self.Nbatch):
            yield np.random.choice(idxAvailable, 
                size = (self.Nbatch,)) # (Nbatch,)

    def getBatchData(self, idx):
# idx: (Nbatch,)
        idxWithSeq = idx + np.arange(1-self.Nseq, 1).reshape(-1, 1)
# (Nseq, Nbatch)
        eventDataBatch = self.eventDataSet.getSlice(idxWithSeq) 
        # (Nseq, Nbatch, Ndelta)
        pvDataBatch = self.pvDataSet.getSlice(idxWithSeq) 
        # (Nseq, Nbatch, Npv)
        return eventDataBatch, pvDataBatch

    def getTrainData(self):
        idxAvailable = self.getAvailableIndex() # (almost N-Nseq+1,)
        return self.getBatchData(idxAvailable) 


