
import itertools

from adapter import EnvironmentAdapter, AgentAdapter, TrainerAdapter
from builder import Builder
from environmentFromDb import EnvironmentFactoryFromDb
from agentGru import AgentFactoryGru
from trainerMLE import TrainerMLEFactory

class Builder001(Builder):
    
    def __init__(self, history):
        super().__init__(history)

    def iterate(self):

        def genBuildOrder():
           
            buildOrder = dict(         
                dbFilePath = "testDb.sqlite"
                , period = ("2020-06-12 17:05:00", "2020-06-13 14:45:00")
                , pv_tags = ("PV0006", "PV0016")
                , pv_preprocesses = (
                    ("None", "Preprocess0001")
                    , ("Preprocess0002",)
                    )
                , ev_tags = ("EV0004", "EV0006")
                , Nbatch = 2**5
                , Nseq = 2**3
                , Nh = 2**5
                , Nepoch = 2**3
                )

            for _ in range(3):
                yield buildOrder

        agentConstructParameterNames = ("Nh",)
        environmentConstructParameterNames = [*map(lambda xx: xx.strip(),
            "dbFilePath, period, pv_tags, pv_preprocesses, ev_tags, Nbatch, Nseq".split(","))] 
        trainerConstructParameterNames = ("Nepoch",)
        
        for buildOrder in genBuildOrder():

            constructorParameter = {key: buildOrder[key] for key in environmentConstructParameterNames}           
            environment = EnvironmentFactoryFromDb().initialize("DataFeederFromDb", **constructorParameter)
            
            environmentAdapter = EnvironmentAdapter(environment)
            environmentAdapter.setConstructorParameter(constructorParameter)

            constructorParameter = dict(
                Ndelta = environment.getNdelta() 
                , Npv = environment.getNpv() 
                )
            constructorParameter = {**constructorParameter, **{
                key: buildOrder[key] for key in agentConstructParameterNames}}

            agent = AgentFactoryGru().create("AgentGru", constructorParameter, saveFilePath = None)
            agentAdapter = AgentAdapter(agent)
            agentAdapter.setConstructorParameter(constructorParameter)

            constructorParameter = {key: buildOrder[key] for key in trainerConstructParameterNames}
            trainer = TrainerMLEFactory().create("TrainerMLE", agent, environment, 
                constructorParameter, None, None)
            trainerAdapter = TrainerAdapter(trainer)
            trainerAdapter.setConstructorParameter(constructorParameter)

            yield agentAdapter, environmentAdapter, trainerAdapter

