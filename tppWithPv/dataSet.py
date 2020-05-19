
import numpy as np


class TestEventDataSet():

    _uniqueInstance = {}

    def __init__(self, dataFilePath):
        
        
        with open(dataFilePath) as fp:
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
    def getInstance(cls, dataFilePath):
        if not dataFilePath in cls._uniqueInstance:
            cls._uniqueInstance[dataFilePath] = super().__new__(cls)
            cls._uniqueInstance[dataFilePath].__init__()
        return cls._uniqueInstance[dataFilePath]

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

    def __init__(self, dataFilePath):
        
        dataRaw = self.readDataRaw(dataFilePath) # (nSample, nPv)
        
        data = (dataRaw - np.nanmean(dataRaw, axis=0))/np.nanstd(dataRaw, axis=0) # (*, Npv)
        
        self.data = data # (Nsample, Npv)

    def readDataRaw(self, dataFilePath):
        
        with open(dataFilePath) as fp:
            txt = fp.read()

        dataRaw = []
        
        # the first line is supposed to be the header line.
        for line in txt.split("\n")[1:]:
            if len(line.rstrip()) > 0:
                cells = line.rstrip().split(",") # each cells contain 0 or 1, which represent the occurrence of a relative event.
                dataRaw.append([float(cell) for cell in cells])
        dataRaw = np.array(dataRaw, dtype=np.float32) # (*, Npv)

        (Nsample, Npv) = dataRaw.shape
        print("A data set with the dimension (Nsample=%d, Npv=%d) has been loaded, successfully." % (Nsample, Npv))
        
        return dataRaw # (nSample, nPv)

    @classmethod
    def getInstance(cls, dataFilePath):
        if not dataFilePath in cls._uniqueInstance:
            cls._uniqueInstance[dataFilePath] = super().__new__(cls)
            cls._uniqueInstance[dataFilePath].__init__()
        return cls._uniqueInstance[dataFilePath]

    def getNsample(self):
        return self.data.shape[0]
    
    def getNpv(self):
        return self.data.shape[1]

    def getSlice(self, idx):
# idx: (...)
        return self.data[idx, :] # (..., Npv)

    def getAvailableIndex(self):
        idxAvailable = np.where(~np.any(np.isnan(self.data), axis=1))[0] # (*,)
        return idxAvailable


class TestPvDataSetWithDifferential(TestPvDataSet):
    
    def __init__(self, dataFilePath, tau):
        super(TestPvDataSetWithDifferential, self).__init__(dataFilePath)
        
        dataRaw = self.readDataRaw(dataFilePath) # (nSample, nTag)
        
        dataRawDiff = TestPvDataSetWithDifferential.makeDifferential(dataRaw, tau) # (nSample, nTag)
        
        dataDiff = (dataRawDiff - np.nanmean(dataRawDiff, axis=0))/np.nanstd(dataRawDiff, axis=0) # (nSample, nTag)
        
        absDataDiff = np.abs(dataDiff) # (nSample, nTag)
        
        self.data = absDataDiff # (nSample, nTag)
        
        
    @classmethod
    def makeDifferential(cls, data, tau):
        # data: (nSample, nTag)
        # tau: integer, tau > 0
        nSample, nTag = data.shape
        dataDiff = np.zeros((nSample, nTag)) # (nSample, nTag)
        dataDiff[:tau-1,:] = np.nan # (tau-1, nTag)
        
        t = np.linspace(-0.5,0.5,tau).reshape(-1,1) # (tau,1)
        tt = np.sum(t*t) # (,)
        for i in range(tau-1, nSample):
            xx = data[i-(tau-1):(i+1), : ] # (tau, nTag)
            xx -= np.nanmean(xx, axis=0) # (tau, nTag)
            diff = np.sum(t * xx, axis=0)/tt # (nTag,)
            dataDiff[i,:] = diff
        
        return dataDiff # (nSample, nTag)
    
    
    