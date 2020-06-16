
import sqlite3
import traceback

from dataSet import DataSetFromDb
from environment import IEnvironment
from factory import EnvironmentFactory
import numpy as np


class EnvironmentFactoryFromDb(EnvironmentFactory):

    def initialize(self, classname, **args):
        return eval(classname)(**args)


class DataFeederFromDb(IEnvironment):

    def __init__(self, dbFilePath, period, pv_tags, pv_preprocesses, ev_tags, Nbatch, Nseq):
        super().__init__()

        sql = """
Select 
    Tag.name
    From Tag
    Where Tag.id in (Select
        p.tag_id
        From PreprocessedTag p
            Join PreprocessMethod p2
                On p.preprocess_method_id = p2.id
            Join Tag 
                On Tag.id = p.source_tag_id
        Where Tag.name = ?
        And p2.name = ? 
        Limit 1)
    Limit 1
"""

        pv_tags_all = None
        try:
            conn = sqlite3.connect(dbFilePath, detect_types = sqlite3.PARSE_COLNAMES|sqlite3.PARSE_DECLTYPES)
            cur = conn.cursor()
            pv_tags_all = []
            for source_tag, preprocesses in zip(pv_tags, pv_preprocesses):
                for preprocess in preprocesses:
                    if preprocess == "None":
                        pv_tags_all.append(source_tag)
                    else:
                        cur.execute(sql, (source_tag, preprocess,))
                        tag, = cur.fetchone()
                        pv_tags_all.append(tag)
        except:
            traceback.print_exc()
            pv_tags_all = None
        finally:
            conn.close()
            
        assert pv_tags_all is not None, "FAILED TO LOADING DATA FROM THE GIVEN DB: %s" % dbFilePath
        
        all_tags = list(ev_tags) + list(pv_tags_all)
        self.ev_idx = [k1 for k1 in range(len(ev_tags))]
        self.pv_idx = [k1 + len(ev_tags) for k1 in range(len(pv_tags_all))]
        self.dataSet = DataSetFromDb.getInstance(dbFilePath=dbFilePath , tags=all_tags, period=period)
        self.Nbatch = Nbatch
        self.Nseq = Nseq
        
        assert self.dataSet.getNsample() > Nseq

    def getAvailableIndex(self):
        idxAvailablePrimitive = self.dataSet.getAvailableIndex()
        idxAvailable = [ i for i in idxAvailablePrimitive
            if np.all(np.isin(np.arange(i+1-self.Nseq, i+1), idxAvailablePrimitive)) ]
        return idxAvailable # (almost N-Nseq+1,)

    def iterate(self):
        idxAvailable = self.getAvailableIndex() # (almost N-Nseq+1,)
        Navailable = len(idxAvailable)
        for _ in range(Navailable//self.Nbatch):
            yield np.random.choice(idxAvailable, 
                size = (self.Nbatch,)) # (Nbatch,)

    def getBatchData(self, idx):
# idx: (Nbatch,)
        idxWithSeq = idx + np.arange(1-self.Nseq, 1).reshape(-1, 1)
# (Nseq, Nbatch)
        dataBatch= self.dataSet.getSlice(idxWithSeq) 
        # (Nseq, Nbatch, Ndelta + nPv)
        
        eventDataBatch = dataBatch[..., self.ev_idx] # (nSeq, nBatch, Ndelta)
        pvDataBatch = dataBatch[..., self.pv_idx] # (nSeq, nBatch, nPv)
        
        return eventDataBatch, pvDataBatch

    def getTrainData(self):
        idxAvailable = self.getAvailableIndex() # (almost N-Nseq+1,)
        return self.getBatchData(idxAvailable) 

    def getNpv(self):
        return len(self.pv_idx)

    def getNdelta(self):
        return len(self.ev_idx)
