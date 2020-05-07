'''
Created on 2020/05/07

@author: ukai
'''
import unittest

from environmentTpp import EventDataFeeder, TestEventDataSet, \
    SingleEventDataFeeder
import numpy as np


class Test(unittest.TestCase):

    def test_001(self):
        import itertools
        Nbatch = 2**5
        Nseq = 12*4
        constructorParameter = dict(Nbatch = Nbatch, Nseq = Nseq)
        dataFeeder = EventDataFeeder(**constructorParameter)
        Ndelta = dataFeeder.Ndelta
        cnt = 0
        itr = dataFeeder.iterate()
        itr = itertools.cycle(itr)
        itr = itertools.islice(itr, 2**10)
        for idx in itr:
            dataBatch = dataFeeder.getBatchData(idx)
            assert dataBatch.shape == (Nseq, Nbatch, Ndelta)
            assert dataBatch.dtype.type is np.float32
            assert not np.any(np.isnan(dataBatch))

            dataTrain = dataFeeder.getTrainData()
            assert dataTrain.shape[0] == Nseq
            assert dataTrain.shape[2] == Ndelta
            cnt += 1
        assert cnt > 0

    @unittest.skip("skip")
    def test_002(self):
        import matplotlib.pylab as plt
        ds = TestEventDataSet.getInstance(Ndelta = 2**0)
        idx = np.arange(0, 2**6)
        data = ds.getSlice(idx)
        plt.plot(data, 'o-')
        plt.show()

    def test_003(self):
        Nbatch = 2**5
        Nseq = 12*4
        Ndelta = 1
        constructorParameter = dict(Nbatch = Nbatch, Nseq = Nseq)
        dataFeeder = SingleEventDataFeeder(**constructorParameter)

        for idx in dataFeeder.iterate():
            dataBatch = dataFeeder.getBatchData(idx) # (Nseq, Nbatch, Ndelta)
            assert dataBatch.shape == (Nseq, Nbatch, Ndelta)
            assert dataBatch.dtype.type is np.float32
            assert np.all(dataBatch[0, :, :] == 1)

            dataTrain = dataFeeder.getTrainData()
            assert dataTrain.shape[0] == Nseq
            assert dataTrain.shape[2] == Ndelta
            assert np.all(dataTrain[0, :, :] == 1)

    def test_004(self):
        ds = TestEventDataSet.getInstance(Ndelta = 2**0)
        assert ds.getNtag() > 0
        Navailable = len(ds.getAvailableIndex())
        assert Navailable <= ds.getNsample()
        assert Navailable > 0


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()