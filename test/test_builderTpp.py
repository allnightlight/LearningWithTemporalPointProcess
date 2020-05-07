'''
Created on 2020/05/07

@author: ukai
'''
import unittest

import torch

from agentTpp import AgentFactoryTpp
from builderTpp import BuilderTpp
from environmentTpp import EnvironmentFactoryTpp
from history import History
from loaderTpp import LoaderTpp
import numpy as np
from trainerTpp import TrainerFactoryTpp
import util
import matplotlib.pylab as plt

class Test(unittest.TestCase):


    def test001(self):
        
        history = History.getInstance("historyTpp.sqlite")
        builder = BuilderTpp(history)
        rtn = builder.build()
        print("agentId, environmentId, trainerId")
        for agentId, environmentId, trainerId in rtn:
            print(agentId, environmentId, trainerId)

        pass
    
    def test002(self):
        
        
        history = History.getInstance("historyTpp.sqlite")
        loader = LoaderTpp(history, AgentFactoryTpp(), EnvironmentFactoryTpp(), 
            TrainerFactoryTpp())
    
        for agent, environment, trainer in loader.iterateHistory():
    
            dataTrain = environment.getTrainData()
            _I = agent(torch.tensor(dataTrain))
            I = _I.data.numpy() # (Nseq+1, *, Ndelta)
    
            for idx in np.random.randint(dataTrain.shape[1], size=(3,)):
    
                F = dataTrain[:, idx, :] # (Nseq, Ndelta)
                B = I[:-1, idx, :] # (Nseq, Ndelta)
    
                Nseq = F.shape[0]
                Ndelta = F.shape[-1]
    
                myXticklabel = [*range(Nseq)]
                myYticklabel = [*range(Ndelta)]
                myTitle = "Test"
    
                fig = plt.figure()
                ax = util.drawColorMapOfEventProb(F, B, myXticklabel, myYticklabel, 
                    myTitle)
                plt.savefig("./tmp/fig.png")
                plt.close(fig)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()