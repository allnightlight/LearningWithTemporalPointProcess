'''
Created on 2020/06/13

@author: ukai
'''
import unittest
from dataSet import DataSetFromDb


class Test(unittest.TestCase):


    def test001(self):
        
        tags = ["PV%04d" % (k1+1)  for k1 in range(10)]
        period = ('2020-06-12 16:20:00', '2020-06-12 17:05:00')
        
        DataSetFromDb("./testDb.sqlite", tags, period)
        
        args = dict(
            dbFilePath = "./testDb.sqlite" 
            , tags = tags
            , period = period
            )
        
        DataSetFromDb.getInstance(**args)
        DataSetFromDb.getInstance(**args)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()