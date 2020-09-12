'''
Created on 17 aot 2017

@author: mamitiana
'''
import os


class CrashHandler(object):
    '''
    classdocs
    '''


    def __init__(self, modelName,folderpath="./"):
        '''
        Constructor
        '''
        
        self.folderpath= folderpath
        self.modelName = modelName
        self.modelOnFolder()
    def modelOnFolder(self):
        literel=[]
        for i in os.listdir(self.folderpath):
            if  self.modelName in i:
                iter=int( i.split('.')[0].split('_')[1] )
                literel.append(iter)
               
        self.literel=sorted(literel)
    def modelPath(self,iter):
        return self.modelName+"_"+str(iter)+".pymodel" 
    
    def latestModel(self):
        if len(self.literel) == 0:
            return None , -1
        return self.modelPath(self.literel[-1] ) , self.literel[-1]
if __name__=="__main__":
    c= CrashHandler('modelResNet',"../")
    print("list iter: "+str(c.literel))
    print("last iter: "+str(c.latestModel()))