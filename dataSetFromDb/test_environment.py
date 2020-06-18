'''
Created on 2020/06/14

@author: ukai
'''
from datetime import datetime
import unittest

from environmentFromDb import EnvironmentFactoryFromDb


class Test(unittest.TestCase):


    def test001(self):

        args = dict(         
            dbFilePath = "./testDb.sqlite"
            , period_train = (
                datetime.strptime("2020/6/12 17:05", "%Y/%m/%d %H:%M")
                , datetime.strptime("2020/6/13 14:45", "%Y/%m/%d %H:%M")
                )
            , period_test= (
                datetime.strptime("2020/6/13 17:05", "%Y/%m/%d %H:%M")
                , datetime.strptime("2020/6/14 14:45", "%Y/%m/%d %H:%M")
                ) 
            , pv_tags = ("PV0006", "PV0016")
            , pv_preprocesses = (
                ("None", "Preprocess0001")
                , ("Preprocess0002",)
                )
            , ev_tags = ("EV0004", "EV0006")
            , Nbatch = 2**5
            , Nseq = 2**3
            , samplingIntervalMinute = 10
            )
        
        factory = EnvironmentFactoryFromDb()
        environment = factory.initialize("DataFeederFromDb", **args)
              
        for idx in environment.iterate():
            E, Pv = environment.getBatchData(idx) # (Nseq, *, Ndelta)

        E, Pv = environment.getTrainData()
        E, Pv = environment.getTestData()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()