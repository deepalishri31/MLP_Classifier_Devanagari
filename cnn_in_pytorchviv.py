

from torch.nn import Module 
from torch.nn import Conv2d 
from torch.nn import Linear 
from torch.nn import MaxPool2d 
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.optim import SGD
from torch.nn import Softmax
from torch.nn import Sigmoid
import pickle
from torch import flatten 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES']='2'

class LeNet(Module):
  def __init__(self,classes,channel=[1,32,64,128],flatten_lay=32*64,pr=0.2): #numChannels=1 if grey image 3 if RGB image
    super(LeNet,self).__init__() 
    self.conv1=Conv2d(in_channels=channel[0],out_channels=channel[1],
                      kernel_size=(5,5),
                      stride=(1,1),
                      padding=2)
    self.relu1=ReLU()
    self.maxpool1=MaxPool2d(kernel_size=(2,2),stride=(2,2))
    self.conv2=Conv2d(in_channels=channel[1],out_channels=channel[2],
                      kernel_size=(5,5),
                      stride=(1,1),
                      padding=2)
    self.relu2=ReLU()
    self.maxpool2=MaxPool2d(kernel_size=(2,2),stride=(2,2))
    self.conv3=Conv2d(in_channels=channel[2],out_channels=channel[3],
                      kernel_size=(5,5),
                      stride=(1,1),
                      padding=2)
    self.fc1=Linear(in_features=channel[3]*64,out_features=flatten_lay)
    self.relu3=ReLU()
    self.fc2=Linear(in_features=flatten_lay,out_features=classes)
    self.Softmax=Softmax(dim=1)
    self.dropout=nn.Dropout(pr)
  def forward(self,x):
    x=self.conv1(x)
    x=self.relu1(x)
    x=self.maxpool1(x)
    x=self.conv2(x)
    x=self.relu2(x)
    x=self.maxpool2(x)
    x=self.conv3(x)
    x=flatten(x,1)
    x=self.fc1(x)
    x=self.relu3(x)
    x=self.dropout(x)
    x=self.fc2(x)
    output=self.Softmax(x)
    return output

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader,TensorDataset
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
from IPython.display import clear_output

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv("/home/divya/vivek5/data.csv")
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
en=OneHotEncoder(handle_unknown = 'ignore')
y=en.fit_transform(y.to_numpy().reshape((-1,1))).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42,shuffle=True,stratify=y)

X_train=X_train.to_numpy().reshape(-1,1,32,32)
X_test=X_test.to_numpy().reshape(-1,1,32,32)

#device using for training
DEVICE='cuda' if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))

trainX=torch.from_numpy(X_train).float()
trainY=torch.from_numpy(y_train).float()
testX=torch.from_numpy(X_test).float()
testY=torch.from_numpy(y_test).float()

train_data=TensorDataset(trainX,trainY)=[]
test_data=TensorDataset(testX,testY)

lossFn=nn.CrossEntropyLoss()
dropout_list=[0,0.1,0.2,0.3,0.4,0.5,0.6]
data=[([1,32,64,128],16*64)]
learn_data=[1e-2]
regularization_list=[1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]
p=0
result={}
for j in dropout_list:
  result[j]={}
  for i in regularization_list:
    BATCH_SIZE=128
    EPOCHS=200
    LR=learn_data[0]
    model=LeNet(classes=36,channel=data[0][0],flatten_lay=data[0][1],pr=j).to(DEVICE)
    opt=SGD(model.parameters(),lr=LR,weight_decay=i)
    trainLoss_list=[]
    trainAcc_list=[]
    testLoss_list=[]
    test_Acc_list=[]
    for epoch in tqdm(range(0,EPOCHS)):
      trainLoss=0
      trainAcc=0
      testLoss=0
      testAcc=0
      samples=0
      model.train() #Calling the train() method of the PyTorch model is required for the model parameters to be updated during backpropagation.
      trainDataLoader=DataLoader(train_data,shuffle=True,batch_size=BATCH_SIZE)
      for (batchX,batchY) in trainDataLoader:
        (batchX,batchY)=(batchX.to(DEVICE),batchY.to(DEVICE))
        predictions=model(batchX)
        loss=lossFn(predictions,batchY) #calculating loss
        #training start
        opt.zero_grad() 
        loss.backward() 
        opt.step() 
        trainLoss+=loss.item() * batchY.size(0)
        trainAcc += (predictions.max(1)[1]==batchY.max(1)[1]).sum().item()
        samples+=batchY.size(0)
      trainLoss_list.append(trainLoss/trainX.shape[0])
      trainAcc_list.append(trainAcc/trainX.shape[0])
      testDataLoader=DataLoader(test_data,shuffle=True,batch_size=BATCH_SIZE)
      samples1=0
      for (batchX,batchY) in testDataLoader:
        (batchX,batchY)=(batchX.to(DEVICE),batchY.to(DEVICE))
        predictions=model(batchX)
        loss=lossFn(predictions,batchY) #calculating loss
        model.eval()
        torch.no_grad()
        testLoss+=loss.item() * batchY.size(0)
        testAcc += (predictions.max(1)[1]==batchY.max(1)[1]).sum().item()
        samples1+=batchY.size(0)
      testLoss_list.append(testLoss/testX.shape[0])
      test_Acc_list.append(testAcc/testX.shape[0])
      clear_output()
    result[j][i]=[trainAcc_list[-1],test_Acc_list[-1],trainLoss_list[-1],testLoss_list[-1]]
    with open("/home/divya/vivek5/accuracy_loss_cnn_lr_1e-2_([1,32,64,128],16*64).pkl",'wb') as file:
        pickle.dump(result,file)