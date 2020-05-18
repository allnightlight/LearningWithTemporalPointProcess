'''
Created on 2020/05/18

@author: ukai
'''
import unittest
import numpy as np
from tppWithPv.dataSet import TestEventDataSet, TestPvDataSet,\
    TestPvDataSetWithDifferential


class Test(unittest.TestCase):

    def test001(self):
        
        for _ in range(2**7):
            nSample, nDelta = np.random.randint(1, 2**5, size=(2,))
            dataRaw = np.random.randint(2, size=(nSample, nDelta))
            filePath = "./dataDelta.csv"
            with open(filePath, "w") as fp:
                fp.write("#\n")
                for row in dataRaw:                
                    fp.write(",".join([str(cell) for cell in row]) + "\n")
            
            ds = TestEventDataSet()
            
            assert ds.getNdelta() == nDelta
            assert ds.getNsample() == nSample
        
    def test002(self):

        for _ in range(2**7):        
            nSample, nPv = np.random.randint(2, 2**5, size=(2,))
            dataRaw = np.random.randn(nSample, nPv)
            filePath = "./dataPv.csv"
            with open(filePath, "w") as fp:
                fp.write("#\n")
                for row in dataRaw:                
                    fp.write(",".join([str(cell) for cell in row]) + "\n")
            
            ds = TestPvDataSet()
            
            assert ds.getNpv() == nPv
            assert ds.getNsample() == nSample
            
            
    
    def test003(self):
        
        for _ in range(2**7):
            
            tau = np.random.randint(2, 30)
            
            nX = np.random.randint(1, 10)
            nT = np.random.randint(tau+1, 2**10)
            
            data = np.random.randn(nT, nX)
            
            dataDiff = TestPvDataSetWithDifferential.makeDifferential(data, tau)
            
            assert dataDiff.shape == (nT, nX)
            assert np.all(np.isnan(dataDiff[:tau-1,:]))
            assert not np.any(np.isnan(dataDiff[tau-1:,:]))
            
            
        tau = 10        
        data = np.linspace(0,1,tau).reshape(-1,1) # (tau, 1)
        dataDiff = TestPvDataSetWithDifferential.makeDifferential(data, tau)
        assert np.abs(dataDiff[-1,0] - 1.) < 1e-8
            
    
    def test004(self):

        for _ in range(2**7):
            tau = 2      
            nSample, nPv = np.random.randint(3, 2**5, size=(2,))
            dataRaw = np.random.randn(nSample, nPv)
            filePath = "./dataPv.csv"
            with open(filePath, "w") as fp:
                fp.write("#\n")
                for row in dataRaw:                
                    fp.write(",".join([str(cell) for cell in row]) + "\n")
            
            ds = TestPvDataSetWithDifferential(tau)
            
            assert ds.getNpv() == nPv
            assert ds.getNsample() == nSample


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()