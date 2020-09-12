
# coding: utf-8

# In[2]:

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


# In[3]:

import os


# In[4]:

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from PIL import Image

from torchvision import transforms
# In[5]:

import jsonLoader

dtype= torch.cuda.FloatTensor
# In[6]:


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
    
class ImageClefDataset(Dataset):
    """Dataset wrapping images and target labels for ImageClef.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """
    def _transfo2mutli(self,dftemp):
        '''prend toutes le classes de la dataset,et entraine une encodeur pour mapper les classes '''
        classes= dftemp["classes"].values
        classes= set(np.hstack(classes) )
        
        lclasses= classes-set({None})
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit( [ lclasses]) 

    def __init__(self, img_path, img_ext, phase):
        if phase == "train": 
            jsdict=jsonLoader.TrainLoader()

        elif phase =="val":
            jsdict = jsonLoader.ValLoader()
        self.phase=phase
        dftemp=pd.DataFrame(jsdict.filtrer())
        #print (len(dftemp))
        #print(dftemp['imageName'].head())
        print("image path"+img_path)
        assert dftemp['imageName'].apply(lambda x: os.path.isfile(os.path.join( img_path , x + img_ext ))).all(), "Some images referenced in the CSV file were not found"
        
        self._transfo2mutli(dftemp)
        
        self.img_path = img_path
        self.img_ext = img_ext


        self.X_train = dftemp['imageName']
        self.y_train = dftemp['classes']

    def __getitem__(self, index):
        filepath=os.path.join(self.img_path, self.X_train[index] + self.img_ext )
        img = Image.open( filepath )
        img = img.convert('RGB')

        img = data_transforms[self.phase](img)

        templabel= self.y_train[index]
        templabel = [] if templabel is None else templabel
        ensclasses = set( self.mlb.classes_ )
        enstemplab= set(templabel)
        
        toremove = enstemplab - ensclasses
        if len(toremove) >0:
            for t in list(sorted(toremove)):
                templabel.remove(t)
        labelEncoded=self.mlb.transform([ templabel]) [0,:]
        
        label = torch.from_numpy(labelEncoded  ).type(dtype)

        return img, label
    
    def getListLabel(self):
        return self.mlb.classes_
    
    def nombreLabel(self):
        return len(self.getListLabel())

    def __len__(self):
        ''' longeur  '''
        return len(self.X_train.index)


# In[7]:

if __name__ == '__main__':

    imp="/home/mamitiana/imageclef/detection/train/image"
    clef= ImageClefDataset(imp,".jpg",'train')
    for i in range(len(clef)):
        im,lab=clef[4]
        if type(im) is not torch.FloatTensor:
            print(type(im))
    print("fin,all of them are tesor")


# In[ ]:



