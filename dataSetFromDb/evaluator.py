'''
Created on 2020/06/18

@author: ukai
'''
import numpy as np
from loader import Loader
import torch

class Evaluator(Loader):
    '''
    classdocs
    '''
    
    
    def __init__(self, history, agentFactory, environmentFactory, trainerFactory, nTol):
        super(Evaluator, self).__init__(history, agentFactory, environmentFactory, trainerFactory)
        
        self.nTol = nTol
        
    def iterateTrainId(self):
        for trainerId in [1,2,3]:
            yield trainerId
    
    def evaluate(self, filePath):
        
        def toString(count, prop, segment):
            
            header="Segment,Delta#,Rate.TN,Rate.FP,Rate.FN,Rate.TP,Count.TN,Count.FP,Count.FN,Count.TP"
            
            txt = ""
            nDelta = count.shape[1]
            for k1 in range(nDelta):
                row = "%s,%d," % (segment, k1) \
                    + ",".join(["%.2f" % elm for elm in prop[:,k1]]) \
                    + ","\
                    + ",".join(["%d" % elm for elm in count[:,k1]])\
                    + "\n"
                txt += row
                
            return txt, header

        txt = ""        
        for agent, environment, _ in self.iterateHistory():
            Eref, Pv = environment.getTrainData()
            count, prop = self.evaluateAnAgent(agent, Eref, Pv, self.nTol)
            rows, header = toString(count, prop, "train")
            txt += rows
            
            Eref, Pv = environment.getTestData()
            count, prop = self.evaluateAnAgent(agent, Eref, Pv, self.nTol)
            rows, header = toString(count, prop, "test")
            txt += rows

            
        txt = header + "\n" + txt
        
        with open(filePath, "w") as fp:
            fp.write(txt)
            
    
    @classmethod
    def evaluateAnAgent(cls, agent, Eref, Pv, nTol):
        # Eref: (nSeq, *, nDelta)
        # Pv: (nSeq, *, nPv)
        
        _Eref = torch.tensor(Eref.astype(np.float32)) # (Nseq, *, Ndelta)
        _Pv = torch.tensor(Pv.astype(np.float32)) # (Nseq, *, nPv)
        _Eest, _ = agent(_Eref[:-1,...], _Pv[1:,...]) # (Nseq, *, Ndelta)
        Phat = _Eest.detach().numpy() # nSeq, *, nDelta
        
        Eest = np.zeros(Phat.shape) # (nSeq, *, nDelta)
        Eest[Phat > 0.5] = 1.0
        
        count, prop = cls.countFpAndFn(Eref[-1,...], Eest[-1,...], nTol)
        
        return count, prop
        
    
    @classmethod
    def countFpAndFn(cls, Eref, Eest, nTol):
        # Eest, Eref: (*, nDelta), in {0,1}        
        # count: (4, nDelta), (TN, FP, FN, TP)
        # prop: (4, nDelta)
        
        assert Eref.shape == Eest.shape
        nSample, nDelta = Eref.shape
        
        count = np.zeros((4, nDelta))
        rate = np.zeros((4, nDelta)) 
        
        ErefBlurred = Eref[:nSample-nTol,:]
        for k1 in range(nTol):
            ErefBlurred = np.nanmax(np.stack((ErefBlurred, Eref[(k1+1):(nSample-nTol+k1+1),:]), axis=0), axis=0) #  (nSample-nTol, nDelta)
        
        for k1 in range(nDelta):
            idxValid = ~(np.isnan(ErefBlurred[:,k1]) | np.isnan(Eest[:nSample - nTol,k1]))
            count[0,k1] = np.sum((ErefBlurred[idxValid,k1] == 0) & (Eest[:nSample - nTol,:][idxValid,k1] == 0)) # true negative
            count[1,k1] = np.sum((ErefBlurred[idxValid,k1] == 0) & (Eest[:nSample - nTol,:][idxValid,k1] == 1)) # false positive
            count[2,k1] = np.sum((ErefBlurred[idxValid,k1] == 1) & (Eest[:nSample - nTol,:][idxValid,k1] == 0)) # false negative
            count[3,k1] = np.sum((ErefBlurred[idxValid,k1] == 1) & (Eest[:nSample - nTol,:][idxValid,k1] == 1)) # true positive
            
        rate[0:2,:] = count[0:2,:]/(count[0,:] + count[1,:] + 1e-16)
        rate[2:4,:] = count[2:4,:]/(count[2,:] + count[3,:] + 1e-16) 
        
        return count, rate