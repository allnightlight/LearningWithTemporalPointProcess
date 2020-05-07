

from environment import IEnvironment
from factory import EnvironmentFactory
import numpy as np


class EnvironmentFactoryTpp(EnvironmentFactory):

    def initialize(self, classname, **args):
        return eval(classname)(**args)


class TestEventDataSet():

    _uniqueInstance = {}

    def __init__(self, Ndelta):

        self.randomState = np.random.RandomState(seed = 1)

        N = 2**10

        Ninterval = 2**3
        Nmutual = 2**1
        self.data = np.zeros((N, Ndelta)) # (N, Ndelta)

        flag = 0
        cnt = 0
        for k1 in range(N):

            if k1 % Ninterval == Ninterval - 1:
                self.data[k1,:] = 1.
                flag = 1
                cnt = 0

            if (cnt > 0) & (flag == 1):
                self.data[k1,:] = self.randomState.rand() < 1/Nmutual

            if flag == 1 & (cnt < Nmutual):
                cnt += 1
            elif cnt == Nmutual:
                flag = 0
                cnt = 0

        self.data = self.data.astype(np.float32) 
        self.data[N//2,0] = np.nan
        # (N, Ndelta)

    @classmethod
    def getInstance(cls, Ndelta):
        if not Ndelta in cls._uniqueInstance:
            cls._uniqueInstance[Ndelta] = super().__new__(cls)
            cls._uniqueInstance[Ndelta].__init__(Ndelta)
        return cls._uniqueInstance[Ndelta]

    def getNsample(self):
        return self.data.shape[0]

    def getNtag(self):
        return self.data.shape[1]

    def getSlice(self, idx):
# idx: (...)
        return self.data[idx, :] # (..., Ndelta)

    def getPositiveIndex(self):
        idxPositive = np.where(np.any(self.data == 1, axis=1))[0] # (*,)
        return idxPositive

    def getAvailableIndex(self):
        idxAvailable = np.where(~np.any(np.isnan(self.data), axis=1))[0] # (*,)
        return idxAvailable


class EventDataFeeder(IEnvironment):

    def __init__(self, Nbatch, Nseq):
        super().__init__()
        self.Ndelta = 5
        self.dataSet = TestEventDataSet.getInstance(self.Ndelta)
        self.Nbatch = Nbatch
        self.Nseq = Nseq
        N = self.dataSet.getNsample()
        assert N > Nseq

    def iterate(self):
        idxAvailable = self.dataSet.getAvailableIndex()
        idxAvailable = [ i for i in idxAvailable
            if np.all(np.isin(np.arange(i+1-self.Nseq, i+1), idxAvailable)) ]
        Navailable = len(idxAvailable)
        for _ in range(Navailable//self.Nbatch):
            yield np.random.choice(idxAvailable, 
                size = (self.Nbatch,)) # (Nbatch,)

    def getBatchData(self, idx):
# idx: (Nbatch,)
        idxWithSeq = idx + np.arange(1-self.Nseq, 1).reshape(-1, 1)
# (Nseq, Nbatch)
        dataBatch = self.dataSet.getSlice(idxWithSeq) # (Nseq, Nbatch, Ndelta)
        return dataBatch

    def getTrainData(self):
        N = self.dataSet.getNsample()
        idx = np.arange(self.Nseq-1, N) # (N-Nseq+1,)
        dataTrain = self.getBatchData(idx) # (Nseq, N-Nseq+1)
        return dataTrain# (Nseq, N-Nseq+1, Ndelta)


class SingleEventDataFeeder(IEnvironment):

    def __init__(self, Nbatch, Nseq):
        super().__init__()
        assert Nseq > 0
        self.Ndelta = 1
        self.dataSet = TestEventDataSet.getInstance(self.Ndelta)
        self.Nbatch = Nbatch
        self.Nseq = Nseq
        N = self.dataSet.getNsample()
        assert N > Nseq

        idxPositive = self.dataSet.getPositiveIndex()
        self.idxAvailable = idxPositive[idxPositive + Nseq  < N]

    def iterate(self):
        Navailable = len(self.idxAvailable)
        for _ in range(Navailable//self.Nbatch):
            yield np.random.choice(self.idxAvailable, 
                size = (self.Nbatch,)) # (Nbatch,)

    def getBatchData(self, idx):
# idx: (Nbatch,)
        idxWithSeq = idx + np.arange(self.Nseq).reshape(-1, 1)
# (Nseq, Nbatch)
        dataBatch = self.dataSet.getSlice(idxWithSeq) # (Nseq, Nbatch, Ndelta)
        return dataBatch

    def getTrainData(self):
# (Nseq, Navailable)
        dataTrain = self.getBatchData(self.idxAvailable) # (Nseq, Navailable)
        return dataTrain# (Nseq, Navailable, Ndelta)
