'''
Created on 2020/05/07

@author: ukai
'''
from datetime import datetime, timedelta
import unittest

from history import History
from util import myDatetimeFormat


class Test(unittest.TestCase):

    def test_001(self):

        history1 = History.getInstance("./testDB.sqlite")
        history2 = History.getInstance("./testDB.sqlite")
        assert history1 == history2
        assert isinstance(history1, History) 

    def test_002(self):
        history = History.getInstance("./testDB.sqlite")
        history.initializeDB()

    def test_003(self):
        history = History.getInstance("./testDB.sqlite")
        history.initializeDB()
        for k1 in range(10):
            agentId = history.getNewAgentId()
            environmentId = history.getNewEnvironmentId()
            trainerId = history.getNewTrainerId()
            assert agentId == 1
            assert environmentId == 1
            assert trainerId == 1

    def test_004(self):

        history = History.getInstance("./testDB.sqlite")
        history.initializeDB()

        agentId = history.getNewAgentId()
        agentType = "TestAgent"
        constructorParameter = dict(param1 = 123, param2 = 1.23, \
            param3 = datetime.strptime("2020/04/04 16:54", myDatetimeFormat))
        saveFilePath = "./tmp/agent123.pt"
        history.saveAgent(agentId, agentType, constructorParameter, \
            saveFilePath)

# TBC = To Be Checked
        agentTypeTBC, constructorParameterTBC, \
            saveFilePathTBC = history.loadAgent(agentId)

        assert agentType == agentTypeTBC
        assert constructorParameter == constructorParameterTBC
        assert saveFilePath == saveFilePathTBC

    def test_005(self):

        history = History.getInstance("./testDB.sqlite")
        history.initializeDB()

        environmentId = history.getNewEnvironmentId()
        environmentType = "TestEnvironment"
        constructorParameter = dict(param1 = 123, param2 = 1.23, \
            param3 = datetime.strptime("2020/04/04 16:54", myDatetimeFormat))
        history.saveEnvironment(environmentId, environmentType, \
            constructorParameter)

# TBC = To Be Checked
        environmentTypeTBC, constructorParameterTBC = \
            history.loadEnvironment(environmentId)

        assert environmentType == environmentTypeTBC
        assert constructorParameter == constructorParameterTBC

    def test_006(self):

        history = History.getInstance("./testDB.sqlite")
        history.initializeDB()

        agentId = history.getNewAgentId()
        environmentId = history.getNewEnvironmentId()
        trainerId = history.getNewTrainerId()
        trainerType = "TestTrainer"
        constructorParameter = dict(param1 = 123, param2 = 1.23, \
            param3 = datetime.strptime("2020/04/04 16:54", myDatetimeFormat))

        criteriaNames = ["foo", "hoge",]
        t0 = datetime.now()
        t1 = t0 + timedelta(days = 1)
        timestamp = [t0, t1,]
        tbl = [[1.23, 4.56,], [100, 200], ]
        trainLog = (timestamp, tbl)

        history.saveTrainer(trainerId, agentId, environmentId, trainerType, \
            constructorParameter, criteriaNames, trainLog)

# TBC = To Be Checked
        agentIdTBC, environmentIdTBC, trainerTypeTBC, constructorParameterTBC,\
            criteriaNamesTBC, trainLogTBC = history.loadTrainer(trainerId)

        timestampTBC, tblTBC = trainLogTBC

        assert agentId              == agentIdTBC
        assert environmentId        == environmentIdTBC
        assert trainerType          == trainerTypeTBC
        assert constructorParameter == constructorParameterTBC
        assert criteriaNames        == criteriaNamesTBC, str(criteriaNamesTBC)
        assert timestamp            == timestampTBC
        assert tbl                  == tblTBC



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()