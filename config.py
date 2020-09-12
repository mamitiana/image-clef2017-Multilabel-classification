
# coding: utf-8

# In[4]:


#salut salut
import os


# In[5]:


class Configuration(object):
    nombretrain= -1# pour toute la dataset
    nombreVal= -1# pour touts la validation 
    #mine
    '''basePathImages = "/home/mamitiana/imageclef/" # 
    trainImages= os.path.join(basePathImages,"train","image") # , chemin absolue pour dossier train
    valImages = os.path.join(basePathImages,"val/","image") # ,"" chemin absolue pour dosier val
    trainext= ".jpg"
    valext=""
    basePathJson = "/home/mamitiana/g5k_imageclef/data.json/" #/home/nnyhoavy/g5k/data.json/
    trainJson = os.path.join(basePathJson,"train.json")
    valJson= os.path.join(basePathJson,"val.json")    
    
    
    
    #on server
    '''
    basePathImages = "/data/nnyhoavy_1334644/image"
    trainImages= os.path.join(basePathImages,"train") # , chemin absolue pour dossier train
    valImages = os.path.join(basePathImages,"val/") # ,"" chemin absolue pour dosier val
    trainext= ".jpg"
    valext=".jpg"
    basePathJson = "/data/nnyhoavy_1334644/ftResNet50/data.json/" #/home/nnyhoavy/g5k/data.json/
    trainJson = os.path.join(basePathJson,"train.json")
    valJson= os.path.join(basePathJson,"val.json")
    
    
    


# In[ ]:



