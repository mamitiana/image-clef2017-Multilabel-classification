# coding: utf-8

'''
Created on 12 aot 2017

@author: mamitiana
'''
from torch import optim
import torch
from torch.autograd.variable import Variable
from torch.utils.data.dataloader import DataLoader
import config
from datasetSync import ImageClefDataset
from model import Net


        
if __name__ == '__main__':
    clefdset= ImageClefDataset(config.Configuration.trainImages,".jpg",'train')
    train_loader = DataLoader(clefdset,
                          batch_size=123,
                          shuffle=True,
                          num_workers=1, # 1 for CUDA
                          pin_memory=True # CUDA only
                         )
    print("training")
    model = Net(clefdset).cuda()
    optimizer  = optim.Adam(model.parameters())
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    model.train()
    
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print('-------------------------------------------------------------------------------------------------')
        print(batch_idx)
        # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        data, target = Variable(data), Variable(target)
        print("data: "+str(type(data))+"  size: "+str(data.size())+"  target "+str(type(target)))
        optimizer.zero_grad()
        output = model(data)
        print("output "+str(output.size() ))
        
        temptarget=target[:,0,:]
        print("target: "+str((temptarget.size())))
        print(type(output))
        print(output)
        print("----------------------------")
        print(temptarget)
        loss = criterion( output ,temptarget )
        loss.backward()
        optimizer.step()
    
    