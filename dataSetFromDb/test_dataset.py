'''
Created on 2020/06/13

@author: ukai
'''
from datetime import timedelta
import unittest

from dataSet import DataSetFromDb


class Test(unittest.TestCase):


    def test001(self):
        
        tags = ["PV%04d" % (k1+1)  for k1 in range(10)]
        period = ('2020-06-12 16:20:00', '2020-06-12 17:05:00')
        samplingIntervalMinute = 15
        
        DataSetFromDb("./testDb.sqlite", tags, period, samplingIntervalMinute)
        
        args = dict(
            dbFilePath = "./testDb.sqlite" 
            , tags = tags
            , period = period
            , samplingIntervalMinute = samplingIntervalMinute
            )
        
        DataSetFromDb.getInstance(**args)
        ds = DataSetFromDb.getInstance(**args)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()