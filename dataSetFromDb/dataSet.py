
import numpy as np
import sqlite3
import traceback
import json


class DataSetFromDb():

    _uniqueInstance = {}

    def __init__(self, dbFilePath, tags, period, samplingIntervalMinute):
        # tags = ["PV0001", "PV0002", ...]
        # period = (t0, t1), t0, t1 as datetime
        # samplingIntervalMinute as int
        
        print("The data will be loaded from the following DB: %s" % dbFilePath
              + "\n with the given tags: %s" % ",".join(tags)
              + "\n with the given period: %s - %s" % (period[0], period[1])
              + "\n with sampling interval = %s" % samplingIntervalMinute
              )

        # This SQL command will extract the dataset with the given tags: "tags"
        # and with the timestamp between the given time period: "period",
        # omitting the rows which contain NULL in a column or some columns.
        # Note, the order alongside tag does not meet with the given tags,
        # though, it's restored later.
        sql = """
Select 
    d.value
    From DataTable d
        Where d.tag in ({0})
        And timestamp >= ?
        And timestamp < ?
        And Cast(strftime('%M', timestamp) as int) % {1} == 0
    Order by d.timestamp_id, d.tag
""".format(",".join(map(lambda xx: '"%s"' % xx, tags)), samplingIntervalMinute)

        conn = None
        data = None
        try:
            conn = sqlite3.connect(dbFilePath, detect_types = sqlite3.PARSE_COLNAMES|sqlite3.PARSE_DECLTYPES)
            cur = conn.cursor()
            cur.execute(sql, period)        
            data = np.array(cur.fetchall(), dtype=np.float32).reshape(-1, len(tags)) # (nSample, nTag)
        except:
            traceback.print_exc()
            data = None
        finally:
            if conn is not None:                
                conn.close()
        
        assert data is not None, "FAILED TO LOADING DATA FROM THE GIVEN DB: %s" % dbFilePath
        self.data = data
        
        self.idxAvailable = np.where(~np.any(np.isnan(data), axis=-1))[0] 
        
        self.nSample = len(self.idxAvailable)
        assert self.nSample > 0, "NO AVAILABLE SAMPLE EXISTS IN THE GIVEN PAIR OF PERIOD AND TAGS."
        
        self.nTag = len(tags)
        assert self.nTag == self.data.shape[1], "SOME GIVEN TAGS COULD NOT BE FOUND IN THE GIVEN DB."
        
        self.data = data[:, np.argsort(np.argsort(tags))]
        
    @classmethod
    def getInstance(cls, **args):        
        key = json.dumps(args)        
        if not key in cls._uniqueInstance:
            cls._uniqueInstance[key] = super().__new__(cls)
            cls._uniqueInstance[key].__init__(**args)
        return cls._uniqueInstance[key]

    def getNsample(self):
        return self.nSample
    
    def getTag(self):
        return self.nTag

    def getSlice(self, idx):           
# idx: (...)
        return self.data[idx, :] # (..., nTag)

    def getAvailableIndex(self):
        return self.idxAvailable.copy()
    