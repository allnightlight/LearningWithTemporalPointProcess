
import itertools
import sys

from adapter import EnvironmentAdapter, AgentAdapter, TrainerAdapter
from builder import Builder
from tppWithPv.agentTpp import AgentFactoryTpp
from tppWithPv.environmentTpp import EnvironmentFactoryTpp
from tppWithPv.trainerTpp import TrainerFactoryTpp
from tppWithPv.dataSet import TestEventDataSet, TestPvDataSet

sys.path.append('../framework')




class BuilderTpp(Builder):

    def __init__(self, history):
        super().__init__(history)

    def iterate(self):

        def genParameter():
            Nh      = (2**0,)
            Nbatch  = (2**6,)
            Nepoch  = (2**0,)
            Nseq    = (2**5,)

            agentType = ('AgentHawkesWithPv',)
            environmentType = ('EventDataFeederWithPv',)
            trainerType = ('TrainerTppMLE',)

            itr = itertools.product(agentType, environmentType, trainerType, 
                Nh, Nbatch, Nseq, Nepoch)
            itr = itertools.cycle(itr)
            itr = itertools.islice(itr, 3)

            for arg in itr:
                yield arg

        Ndelta = TestEventDataSet.getInstance().getNdelta()
        Npv = TestPvDataSet.getInstance().getNpv()

        for agentType, environmentType, trainerType, Nh, Nbatch, Nseq, Nepoch \
            in genParameter():

            constructorParameter = dict(Nbatch = Nbatch, Nseq = Nseq)
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

