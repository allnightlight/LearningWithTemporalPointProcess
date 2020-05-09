
import numpy as np


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

    def getSlice(self, idx):
# idx: (...)
        return self.data[idx, :] # (..., Ndelta)

    def getAvailableIndex(self):
        idxAvailable = np.where(~np.any(np.isnan(self.data), axis=1))[0] # (*,)
        return idxAvailable


class TestPvDataSet():

    _uniqueInstance = {}

    def __init__(self, Npv):

        self.Npv = Npv

    @classmethod
    def getInstance(cls, Npv):
        if not Npv in cls._uniqueInstance:
            cls._uniqueInstance[Npv] = super().__new__(cls)
            cls._uniqueInstance[Npv].__init__(Npv)
        return cls._uniqueInstance[Npv]

    def getNsample(self):
        N = 2**10
        return N

    def getSlice(self, idx):
# idx: (...)
        pvDataBatch = np.random.randn(*idx.shape, self.Npv) # (..., Npv)
        return pvDataBatch

    def getAvailableIndex(self):
        idxAvailable = np.arange(self.getNsample()) # (*,)
        return idxAvailable # (*,)

