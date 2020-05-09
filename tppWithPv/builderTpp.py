
import sys
sys.path.append('../framework')

import itertools

from adapter import AgentAdapter, EnvironmentAdapter, TrainerAdapter
from agentTpp import AgentFactoryTpp
from builder import Builder
from environmentTpp import EnvironmentFactoryTpp
from history import History
from trainerTpp import TrainerFactoryTpp


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

        Ndelta = 5
        Npv = 3

        for agentType, environmentType, trainerType, Nh, Nbatch, Nseq, Nepoch \
            in genParameter():

            constructorParameter = dict(Nbatch = Nbatch, Nseq = Nseq, 
                Npv = Npv, Ndelta = Ndelta)
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

