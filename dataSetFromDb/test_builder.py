'''
Created on 2020/06/16

@author: ukai
'''
import unittest

from builder001 import Builder001
from history import History


class Test(unittest.TestCase):


    def test001(self):
        
        history = History("./tmp/test.db")
        builder = Builder001(history)
        builder.build()



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()