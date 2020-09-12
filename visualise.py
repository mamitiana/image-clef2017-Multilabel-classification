'''
pour la visualisation d'une ensemble de donnée
on se contentera d'un seul bacth
'''
from torch.utils.data.dataloader import DataLoader
import torchvision

import config
from datasetSync import ImageClefDataset
import numpy as np
import matplotlib.pyplot as plt

def imageShow(dataloader, toshow=-1):
    '''data loader: pour la sampling, donne directement une batch
    toshow= nombre d'image a afficher = -1 default toute la batch
     '''
    inputs, classes = next(iter(dataloader))
    if toshow ==-1:
        toshow=inputs.size()[0]
    inputs,  classes=inputs[:toshow] , classes[:toshow]
    
    out = torchvision.utils.make_grid(inputs)
    inp = out.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    
    classes=classes.cpu().numpy()
    multilabelEncoder= dataloader.dataset.mlb
    
    title = multilabelEncoder.inverse_transform(classes) 
    plt.title(title)
    plt.imshow(inp)
    plt.show()
    
    

    
if __name__=="__main__":
    clefdset = {x: ImageClefDataset(config.Configuration.__dict__[ x+"Images"],".jpg",x)  for x in ['train', 'val']}
    
    
    dset_loaders = {x: DataLoader(clefdset[x], batch_size=128,
                        shuffle=True,
                        num_workers=1, # 1 for CUDA 
                        )for x in ['train', 'val']
                    }
    
    imageShow(dset_loaders['train'], toshow=5)