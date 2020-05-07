'''
Created on 2020/05/07

@author: ukai
'''

from datetime import datetime
import unittest

import numpy as np
import matplotlib.pylab as plt
from util import myDatetimeFormat, json_date_hook, json_converter, convertToJson, \
    parseFromJson, drawColorMapOfEventProb


class Test(unittest.TestCase):


    def test_001(self):
        d = {"dateNow": datetime.strftime(datetime.now(), myDatetimeFormat), 
            "foo": None}
        d = json_date_hook(d)
        assert isinstance(d["dateNow"], datetime)

        t = json_converter(datetime.now())
        assert isinstance(t, str)

        data = {"dateNow": datetime.strptime(datetime.strftime(datetime.now(), 
            myDatetimeFormat), myDatetimeFormat), "foo": None, "float": 10.0, 
            "int": 10}

        dataJson = convertToJson(data)
        dataConverted = parseFromJson(dataJson)

        for key in data:
            assert dataConverted[key] == data[key]

        data = {}
        dataJson = convertToJson(data)
        dataConverted = parseFromJson(dataJson)

    def test_002(self):
        Nx, Ny = 12, 5
        F = np.random.randint(2, size=(Nx, Ny)).astype(np.float) # (Nx, Ny)
        B = np.random.rand(Nx, Ny) # (Nx, Ny)
        myXticklabel = ["label%02d" % k1 if k1 % 5 ==0 else "" 
            for k1 in range(Nx)]
        myYticklabel = ["label%02d" % k2 for k2 in range(Ny)]
        myTitle = "Example"

        fig = plt.figure()
        ax = drawColorMapOfEventProb(F, B, myXticklabel, myYticklabel, myTitle)
        #plt.show()
        plt.close(fig)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()