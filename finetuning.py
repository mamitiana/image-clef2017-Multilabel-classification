'''
Created on 15 ao�t 2017

@author: mamitiana
'''
import copy
import os
import time
from torch import nn, optim
import torch
from torch.autograd.variable import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import models

import config
from crashHandler import CrashHandler
from datasetSync import ImageClefDataset
from evaluation import f1_scoreMultilabs
import numpy as np


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def cretemptyfile(fname):
    with open(fname,'w') as f:
        f.write("")

def writefile(val,fname):
    with open(fname,"a") as f:
        f.write(val)
        f.write("\n")
        
def saveModel(model,path):
    torch.save(model.state_dict(), path)

def loadModel(model,path):
    model.load_state_dict(torch.load(path))
    

def trainModel(model,dset_loaders,optimizer,criterion,lr_scheduler,lastIter,num_epoch,lastepoch):
    since = time.time()
    best_model = model
    best_acc = 0.0
    #training phase
    iteration = lastIter
    print("start iteration from: "+str(iteration))
    print("start epch from: "+str(lastepoch))
    stepsize= 500 #rhf savena n model
    cretemptyfile("historyLoss")
    cretemptyfile("historyF1")
    for epoch in range(lastepoch,num_epoch):
        print("Epoch: "+str(epoch)+" /"+str(num_epoch-1))
        #initisalisation du models selon les cas
        for phase in ['train',"val"]:
            print("Phase: "+str(phase))
            if phase =="train":
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)
            else :model.train(False)        
            curloader = dset_loaders[phase] # getting the data sampeleer
            valF1=0
            for data in curloader:
                input , target = Variable(data[0]) ,Variable(data[1])
                if use_gpu ==True:
                    input , target = Variable(data[0].cuda()) ,Variable(data[1].cuda())
                optimizer.zero_grad()
                outputs = model(input)
               
                loss = criterion(outputs, target)
                outputs.data =outputs.data >-3
                
                writefile(str(loss.data[0]), "historyLoss")
                #historyloss.append(loss.data[0])
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    
                    if iteration%stepsize ==0:
                        print("i: "+str(iteration)+"loss: "+str(loss.data[0]))
                        path= "modelResNet_"+str(iteration)+".pymodel"
                        saveModel(model, path)
                    iteration+=1
                    #print("trainBatchF1: "+str(curF1))
                if phase =="val":
                    curF1=f1_scoreMultilabs(outputs.data.cpu().numpy(),target.data.cpu().numpy())
                    valF1+=curF1
                    print("curvalF1: "+str(curF1))
            if phase == "val":
                writefile(str(valF1), "historyF1")
                print("allvalF1: "+str(valF1))



if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    print("use_gpu:"+str(use_gpu))
    #clefdset= ImageClefDataset(config.Configuration.trainImages,"",'train')
    clefdset = {'train': ImageClefDataset(config.Configuration.trainImages,config.Configuration.trainext,'train') ,
                'val':ImageClefDataset(config.Configuration.valImages,config.Configuration.valext,'val')
                }
    
    clefdset['val'].mlb  = clefdset['train'].mlb
    
    batch_size =50
    dset_loaders = {x: DataLoader(clefdset[x], batch_size=batch_size, shuffle=True ) for x in ['train', 'val'] }
    
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, clefdset['train'].nombreLabel())
    
    crash = CrashHandler('modelResNet')
    lastModelpath , lastIter  = crash.latestModel()
    lastepoch=0
    if lastModelpath is not None:
        loadModel(model_ft,lastModelpath)
        print("lastModel loaded "+str(lastModelpath))
        lastepoch= int(  ( lastIter * batch_size ) // len(clefdset['train']) ) 
    
    if use_gpu:
        model_ft = model_ft.cuda()
    
    optimizer  = optim.SGD(model_ft.parameters(), lr=0.000001, momentum=0.9,weight_decay=0.0005)
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    trainModel(model_ft,dset_loaders,optimizer,criterion,exp_lr_scheduler,lastIter,num_epoch=300,lastepoch=lastepoch)
    