'''
Created on 2020/06/14

@author: ukai
'''
import unittest

from environmentFromDb import EnvironmentFactoryFromDb


class Test(unittest.TestCase):


    def test001(self):

        args = dict(         
            dbFilePath = "./testDb.sqlite"
            , period = ("2020-06-12 17:05:00", "2020-06-13 14:45:00")
            , pv_tags = ("PV0006", "PV0016")
            , pv_preprocesses = (
                ("None", "Preprocess0001")
                , ("Preprocess0002",)
                )
            , ev_tags = ("EV0004", "EV0006")
            , Nbatch = 2**5
            , Nseq = 2**3
            )
        
        factory = EnvironmentFactoryFromDb()
        environment = factory.initialize("DataFeederFromDb", **args)
              
        for idx in environment.iterate():
            E, Pv = environment.getBatchData(idx) # (Nseq, *, Ndelta)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()