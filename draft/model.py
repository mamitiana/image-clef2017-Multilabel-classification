# coding: utf-8

'''
Created on 12 aout 2017
mmmmmmmmmmmm
@author: mamitiana
'''


from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.preprocessing import MultiLabelBinarizer

class Net(nn.Module):
    ''' Network definition '''
    def __init__(self,dataset):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(186624, 4608)
        print("num label: "+str(dataset.nombreLabel()))
        self.fc2 = nn.Linear(4608, dataset.nombreLabel())
        

    def forward(self, x):
        #print(type(x))
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1) # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)