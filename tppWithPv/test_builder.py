'''
Created on 2020/05/19

@author: ukai
'''
import os
import shutil
import unittest

from history import History
import numpy as np
from tppWithPv.builderTpp import BuilderTpp


class Test(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        if not os.path.exists('./tmp'):
            os.mkdir("./tmp")
        
        nSample = 2**10
        nDelta, nPv = np.random.randint(1, 2**2, size=(2,))
        dataRaw = np.random.randint(2, size=(nSample, nDelta))
        eventDataFilePath = BuilderTpp.eventDataFilePath
        with open(eventDataFilePath, "w") as fp:
            for row in dataRaw:                
                fp.write(",".join([str(cell) for cell in row]) + "\n")
                
        dataRaw = np.random.randn(nSample, nPv)
        pvDataFilePath = BuilderTpp.pvDataFilePath
        with open(pvDataFilePath, "w") as fp:
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



    def testName(self):
        history = History("./tmp/test.db")
        builder = BuilderTpp(history)
        builder.build()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()