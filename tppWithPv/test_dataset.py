'''
Created on 2020/05/18

@author: ukai
'''
import unittest
import numpy as np
from tppWithPv.dataSet import TestEventDataSet


class Test(unittest.TestCase):


    def test001(self):
        
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


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()