'''
Created on 2020/05/07

@author: ukai
'''
import unittest

from environment import TestEnvironment, IEnvironment


class Test(unittest.TestCase):


    def test_001(self):
        environment = TestEnvironment()
        assert isinstance(environment, IEnvironment)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()