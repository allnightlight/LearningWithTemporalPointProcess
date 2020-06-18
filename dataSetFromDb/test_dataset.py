'''
Created on 2020/06/13

@author: ukai
'''
from datetime import timedelta, datetime
import unittest

from dataSet import DataSetFromDb


class Test(unittest.TestCase):


    def test001(self):
        
        tags = ["PV%04d" % (k1+1)  for k1 in range(10)]
        period = (datetime.strptime("2020/6/12 16:20", "%Y/%m/%d %H:%M")
                  , datetime.strptime("2020/6/12 17:05", "%Y/%m/%d %H:%M"))
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
        
        timestampAvailable = ds.getAvailableTimestamp()
        assert len(timestampAvailable) > 0
        for s,t in zip(timestampAvailable[:-1], timestampAvailable[1:]):
            dt = t-s
            assert dt.total_seconds() == samplingIntervalMinute * 60
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()