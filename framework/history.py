
from datetime import datetime, timedelta
import os
import sqlite3

from util import convertToJson, parseFromJson, myDatetimeFormat


# <<interface>>
class IHistory:

    def getNewAgentId(self):
        raise NotImplementedError

    def getNewEnvironmentId(self):
        raise NotImplementedError

    def getNewTrainerId(self):
        raise NotImplementedError

    def saveAgent(self, agentId, agentType, constructorParameter, saveFilePath):
        raise NotImplementedError

    def saveEnvironment(self, environmentId, environmentType, 
        constructorParameter):
        raise NotImplementedError

    def saveTrainer(self, trainerId, agentId, environmentId, trainerType,\
        constructorParameter, criteriaNames, trainLog):
        raise NotImplementedError


# <<interface>>
class IHistoryRequiredByLoader:
    def loadAgent(self, agentId):
        raise NotImplementedError
#         return agentType, constructorParameter, saveFilePath

    def loadEnvironment(self, environmentId):
        raise NotImplementedError
#         return environmentType, constructorParameter

    def loadTrainer(self, trainerId):
        raise NotImplementedError
#         return agentId, environmentId, trainerType, constructorParameter, \
#             criteriaNames, trainLog


class TestHistory(IHistory):

    def __init__(self):
        super().__init__()

    def getNewAgentId(self):
        return 0

    def getNewEnvironmentId(self):
        return 1

    def getNewTrainerId(self):
        return 2

    def saveAgent(self, agentId, agentType, constructorParameter, saveFilePath):
        pass

    def saveEnvironment(self, environmentId, environmentType, 
        constructorParameter):
        pass

    def saveTrainer(self, trainerId, agentId, environmentId, trainerType,\
        constructorParameter, criteriaNames, trainLog):
        pass


class History(IHistory, IHistoryRequiredByLoader):

    _uniqueInstance = {}

    def __init__(self, dbFilePath):
        super().__init__()

        alreadyDbExisted = os.path.exists(dbFilePath)

        self.conn = sqlite3.connect(dbFilePath, 
            detect_types = sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        self.cur = self.conn.cursor()

        if not alreadyDbExisted:
            self.initializeDB()

    def __del__(self):
        self.conn.close()

    def initializeDB(self):
        sql = """\

Drop Table If Exists Agent;
Drop Table If Exists Environment;
Drop Table If Exists Trainer;
Drop Table If Exists TrainLog;
Drop Table If Exists CriteriaName;

Create Table Agent (
    id              Integer Primary Key,
    dateUpdate      Timestamp,
    type            Text,
    saveFilePath    Text,
    parameter       Text
    );

Create Table Environment (
    id              Integer Primary Key,
    dateUpdate      Timestamp,
    type            Text,
    parameter       Text
    );

Create Table Trainer (
    id              Integer Primary Key,
    agentId         Integer,
    environmentId   Integer,
    dateUpdate      Timestamp,
    type            Text,
    parameter       Text
    );

Create Table TrainLog (
    id              Integer Primary Key,
    TrainerId       Integer,
    date            timestamp,
    criteriaNameId  Integer,
    value           Real
    );

Create Table CriteriaName (
    id      Integer Primary Key,
    value   Text Unique
    );

"""
        self.cur.executescript(sql)

    @classmethod
    def getInstance(cls, dbFilePath):
        if not dbFilePath in cls._uniqueInstance: 
            cls._uniqueInstance[dbFilePath] = super().__new__(cls)
            cls._uniqueInstance[dbFilePath].__init__(dbFilePath)
        return cls._uniqueInstance[dbFilePath]

    def getNewAgentId(self):
        self.cur.execute("Select count(id) From Agent")
        cnt, = self.cur.fetchone()
        agentId = cnt + 1 # 1,2,3,...
        return agentId

    def getNewEnvironmentId(self):
        self.cur.execute("Select count(id) From Environment")
        cnt, = self.cur.fetchone()
        environmentId = cnt + 1 # 1,2,3,...
        return environmentId

    def getNewTrainerId(self):
        self.cur.execute("Select count(id) From Trainer")
        cnt, = self.cur.fetchone()
        trainerId = cnt + 1 # 1,2,3,...
        return trainerId

    def saveAgent(self, agentId, agentType, constructorParameter, \
        saveFilePath):

        parameterJson = convertToJson(constructorParameter)
        self.cur.execute("""\
            insert into agent (id, dateupdate, type, savefilepath, parameter)
            values (?,?,?,?,?)""", (agentId, datetime.now(), agentType, 
            saveFilePath, parameterJson))
        self.conn.commit()

    def saveEnvironment(self, environmentId, environmentType, \
        constructorParameter):

        parameterJson = convertToJson(constructorParameter)
        self.cur.execute("""\
            Insert Into Environment (id, dateupdate, type, parameter)
            values (?,?,?,?)""", (environmentId, datetime.now(), 
            environmentType, parameterJson))
        self.conn.commit()
        pass

    def saveTrainer(self, trainerId, agentId, environmentId, trainerType,\
        constructorParameter, criteriaNames, trainLog):
        timestamp, tbl = trainLog

        parameterJson = convertToJson(constructorParameter)
        self.cur.execute("""\
            Insert Into Trainer (id, agentId, environmentId, dateupdate, 
            type, parameter) values (?,?,?,?,?,?)""", 
            (trainerId, agentId, environmentId, datetime.now(), 
            trainerType, parameterJson))

        for name in criteriaNames:
            self.cur.execute("""Insert Or Ignore Into CriteriaName 
                (value) values (?)""", (name,))

        criteriaNameIds = []
        for name in criteriaNames:
            self.cur.execute("Select id From CriteriaName Where value = ?", 
                (name,))
            criteriaNameIds.append(self.cur.fetchone()[0])

        for t, arr in zip(timestamp, tbl):
            for criteriaNameId, val in zip(criteriaNameIds, arr):
                self.cur.execute("""\
                Insert Into TrainLog (TrainerId, date, criteriaNameId, value)
                values (?, ?, ?, ?)""", (trainerId, t, criteriaNameId, val))

        self.conn.commit()

    def loadAgent(self, agentId):
        self.cur.execute("""\
            Select type, parameter, saveFilePath 
            From Agent 
            Where id = ?""", (agentId,))
        agentType, constructorParameterJson, saveFilePath = self.cur.fetchone()
        constructorParameter = parseFromJson(constructorParameterJson)
        return agentType, constructorParameter, saveFilePath

    def loadEnvironment(self, environmentId):
        self.cur.execute("""\
            Select type, parameter 
            From Environment
            Where id = ?""", (environmentId,))
        environmentType, constructorParameterJson = self.cur.fetchone()
        constructorParameter = parseFromJson(constructorParameterJson)
        return environmentType, constructorParameter

    def loadTrainer(self, trainerId):
        self.cur.execute("""\
            Select agentId, environmentId,  type, parameter 
            From Trainer
            Where id = ?""", (trainerId,))

        agentId, environmentId, trainerType, constructorParameterJson = \
            self.cur.fetchone()
        constructorParameter = parseFromJson(constructorParameterJson)

        self.cur.execute("""\
            Select Distinct CriteriaName.value From CriteriaName Join TrainLog
                Where CriteriaName.id == TrainLog.criteriaNameId
                And TrainLog.TrainerId == ?
                Order by CriteriaName.id""", (trainerId,))

        criteriaNames = [elm for elm, in self.cur.fetchall()]

        self.cur.execute("""\
            Select CriteriaName.id From CriteriaName Join TrainLog
                Where CriteriaName.id == TrainLog.criteriaNameId
                And TrainLog.TrainerId == ?
                Limit 1""", (trainerId,))
        anyCriteriaId, = self.cur.fetchone()

        self.cur.execute("""\
            Select date From CriteriaName Join TrainLog
                Where CriteriaName.id == TrainLog.criteriaNameId
                And CriteriaName.id == ?
                And TrainLog.TrainerId == ?
                Order by TrainLog.date""", (anyCriteriaId, trainerId,))

        timestamp = [elm for elm, in self.cur.fetchall()]

        self.cur.execute("""\
            Select TrainLog.value From CriteriaName Join TrainLog
                Where CriteriaName.id == TrainLog.criteriaNameId
                And TrainLog.TrainerId == ?
                Order by TrainLog.date, CriteriaName.id""", (trainerId,))

        row = [elm for elm, in self.cur.fetchall()]
        assert len(row) == len(timestamp) * len(criteriaNames)

        tbl = []
        for k1 in range(len(timestamp)):
            tbl.append(row[k1*len(criteriaNames):(k1+1)*len(criteriaNames)])

        trainLog = (timestamp, tbl)

        return agentId, environmentId, trainerType, constructorParameter, \
            criteriaNames, trainLog

