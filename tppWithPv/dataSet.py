
import numpy as np


class TestEventDataSet():

    _uniqueInstance = None

    def __init__(self):
        
        
        with open("dataDelta.csv") as fp:
            txt = fp.read()

        dataRaw = []
        
        # the first line is supposed to be the header line.
        for line in txt.split("\n")[1:]:
            if len(line.rstrip()) > 0:
                cells = line.rstrip().split(",") # each cells contain 0 or 1, which represent the occurrence of a relative event.
                dataRaw.append([float(cell) for cell in cells])
        dataRaw = np.array(dataRaw, dtype=np.float32) # (*, Ndelta)
        
        (Nsample, Ndelta) = dataRaw.shape
        print("A data set with the dimension (Nsample=%d, Ndelta=%d) has been loaded, successfully." % (Nsample, Ndelta))

        self.data = dataRaw # (Nsample, Ndelta)

    @classmethod
    def getInstance(cls):
        if cls._uniqueInstance is None:
            cls._uniqueInstance = super().__new__(cls)
            cls._uniqueInstance.__init__()
        return cls._uniqueInstance

    def getNsample(self):
        return self.data.shape[0]
    
    def getNdelta(self):
        return self.data.shape[1]

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

