
import itertools

from adapter import EnvironmentAdapter, AgentAdapter, TrainerAdapter
from builder import Builder
from tppWithPv.agentTpp import AgentFactoryTpp
from tppWithPv.environmentTpp import EnvironmentFactoryTpp
from tppWithPv.trainerTpp import TrainerFactoryTpp
from tppWithPv.dataSet import TestEventDataSet, TestPvDataSet


class BuilderTpp(Builder):
    
    eventDataFilePath = "./tmp/dataDelta.csv"
    pvDataFilePath = "./tmp/dataPv.csv"

    def __init__(self, history):
        super().__init__(history)

    def iterate(self):

        def genParameter():
            Nh      = (2**0,)
            Nbatch  = (2**6,)
            Nepoch  = (2**0,)
            Nseq    = (2**5,)
            preprocess = ('None', 'Differential',)
            tau = (2**2,)

            agentType = ('AgentHawkesWithPv',)
            environmentType = ('EventDataFeederWithPv',)
            trainerType = ('TrainerTppMLE',)

            itr = itertools.product(agentType, environmentType, trainerType, 
                Nh, Nbatch, Nseq, Nepoch, preprocess, tau)
            itr = itertools.cycle(itr)
            itr = itertools.islice(itr, 3)

            for arg in itr:
                yield arg

        Ndelta = TestEventDataSet.getInstance(BuilderTpp.eventDataFilePath).getNdelta()
        Npv = TestPvDataSet.getInstance(BuilderTpp.pvDataFilePath).getNpv()

        for agentType, environmentType, trainerType, Nh, Nbatch, Nseq, Nepoch, preprocess, tau\
            in genParameter():

            constructorParameter = dict(Nbatch = Nbatch, Nseq = Nseq, preprocess = preprocess, tau = tau, eventDataFilePath = BuilderTpp.eventDataFilePath, pvDataFilePath = BuilderTpp.pvDataFilePath)
            environment = EnvironmentFactoryTpp().create(environmentType, 
                constructorParameter)
            environmentAdapter = EnvironmentAdapter(environment)
            environmentAdapter.setConstructorParameter(constructorParameter)

            constructorParameter = dict(Ndelta = Ndelta, Nh = Nh, Npv = Npv)
            agent = AgentFactoryTpp().create(agentType, constructorParameter, 
                saveFilePath = None)
            agentAdapter = AgentAdapter(agent)
            agentAdapter.setConstructorParameter(constructorParameter)

            constructorParameter = dict(Nepoch = Nepoch)
            trainer = TrainerFactoryTpp().create(trainerType, agent, environment, 
                constructorParameter, None, None)
            trainerAdapter = TrainerAdapter(trainer)
            trainerAdapter.setConstructorParameter(constructorParameter)

            yield agentAdapter, environmentAdapter, trainerAdapter

