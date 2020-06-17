'''
Created on 2020/06/15

@author: ukai
'''
import unittest

from agentGru import AgentGru
from environmentFromDb import EnvironmentFactoryFromDb
from trainerMLE import TrainerMLE


class Test(unittest.TestCase):


    def test001(self):
        
        Nh = 2**5
        Nepoch = 2**3
        
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
            , samplingIntervalMinute = 10
            )

        factory = EnvironmentFactoryFromDb()
        environment = factory.initialize("DataFeederFromDb", **args)
        
        args = dict(
            Ndelta = environment.getNdelta()
            , Npv = environment.getNpv()
            , Nh = Nh)
        
        agent = AgentGru(**args)
        
        trainer = TrainerMLE(agent, environment, Nepoch)
        
        trainer.train()
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()