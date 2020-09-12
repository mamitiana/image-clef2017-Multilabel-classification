
# coding: utf-8

# In[2]:

import config
import json as js


# In[3]:

class JsonLoader(object):
    ''' base -chargement de json '''
    def __init__(self,path,limit ):
        self.path = path
        self.limit = limit
        self._loadJson()
    def filtrer(self):
        ''' filtrer le nombre de image a utiliser '''
        temp=self.limit
        return self.content[:temp] if temp >0 else self.content
    
    def _loadJson(self):
        f=open(self.path)
        jsonstr=f.read()
        self.content=js.loads(jsonstr)  


# In[4]:

class TrainLoader(JsonLoader):
    ''' chargement du json de train ,selon  config pour train'''
    def __init__(self):
        JsonLoader.__init__(self,config.Configuration.trainJson,config.Configuration.nombretrain)


# In[5]:

class ValLoader(JsonLoader):
    ''' chargement du json de train ,selon  config pour validation'''
    def __init__(self):
        JsonLoader.__init__(self,config.Configuration.valJson,config.Configuration.nombreVal)


# In[6]:

from sklearn.preprocessing import MultiLabelBinarizer


# In[8]:

if __name__=="__main__":
    valjs = TrainLoader()
  


# In[7]:

''


# In[21]:

len(set([3,32,4]))


# In[ ]:



