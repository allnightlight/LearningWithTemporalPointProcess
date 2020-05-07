'''
Created on 2020/05/07

@author: ukai
'''
import unittest

from builder import Builder, TestBuilder
from history import History


class Test(unittest.TestCase):

    def test_001(self):
        history = History.getInstance("testDB.sqlite")        
        builder = TestBuilder(history)
        
        assert isinstance(builder, Builder)
        rtn = builder.build()




if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()