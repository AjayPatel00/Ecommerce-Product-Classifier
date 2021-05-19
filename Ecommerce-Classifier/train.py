import torch as t, torch.nn as nn, torch.nn.functional as tnnF
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision as tv, torchvision.transforms as tr
from torchvision import models
from collections import Counter
import matplotlib.pyplot as plt
import pdb
import os
import sys
import numpy as np
n_classes = 27
seed = 1
t.manual_seed(seed)
if t.cuda.is_available(): t.cuda.manual_seed_all(seed)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight   

def evaluate(f,data,device):
    f.eval()
    losses,accs = [],[]
    for x,y in data:
        x,y = x.to(device),y.to(device)
        logits = f(x)
        loss = nn.CrossEntropyLoss()(logits,y)
        acc = (logits.max(1)[1] == y).float().mean()
        losses.append(loss.item())
        accs.append(acc.item())
    f.train()
    return np.mean(losses),np.mean(accs)

def load_data(path,preloaded=True):
    if not preloaded:
        im_sz = 64
        RGB_MEAN = [0.8235, 0.8086, 0.8016]
        RGB_STD = [0.2145, 0.2261, 0.2292]
        fn = tr.Compose([tr.Pad(4,padding_mode='reflect'),
                        tr.Resize(im_sz),
                        tr.CenterCrop(im_sz),
                        #tr.GaussianBlur(5,1),
                        #tr.Grayscale(),
                        tr.ToTensor(),
                        tr.RandomHorizontalFlip()])
                        #tr.Normalize(mean=RGB_MEAN,std=RGB_STD)])
        #dset = tv.datasets.ImageFolder(root='/content/drive/MyDrive/CS480/Final/train',transform=fn)
        dset = tv.datasets.ImageFolder(root=path,transform=fn)
        lbls = [dset[i][1] for i in range(len(dset))]
        cnts = Counter(lbls)
        pdb.set_trace()
        print(dset[0][1])
    else:
        loaded = np.load("PATH",allow_pickle=True)
        x,y = t.Tensor([_[0] for _ in loaded]), t.Tensor([_[1] for _ in loaded])
        dset = TensorDataset(x,y)

    lbls = [dset[i][1] for i in range(len(dset))]
    cnts = Counter(lbls)

    test_ctrs = [int(0.1*v) for v in cnts.values()] # then try [5]*27
    train_inds,test_inds = [],[]
    for i,(x,y) in enumerate(dset):
        if test_ctrs[y] > 0:
            test_inds.append(i)
            test_ctrs[y] -= 1
        else:
            train_inds.append(i)

    dset_train = Subset(dset,train_inds)
    imgs = [d[0] for d in dset_train]
    dset_test = Subset(dset,test_inds)

    weights = make_weights_for_balanced_classes(dset_train,n_classes)
    weights = t.DoubleTensor(weights)
    sampler = t.utils.data.sampler.WeightedRandomSampler(weights,len(weights))

    dload_train = DataLoader(dset_train,sampler=sampler,batch_size=32,num_workers=4,drop_last=True)
    dload_test = DataLoader(dset_test,batch_size=32,num_workers=4,drop_last=True)
    return dload_train,dload_test

class F(nn.Module):
    def __init__(self,model_arch="resnet18",n_classes=27):
        super(F,self).__init__()
        if model_arch == "resnet18":
            self.model = models.resnet18()#pretrained=True)
            self.num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(self.num_ftrs,n_classes)

    def forward(self,x):
        logits = self.model(x)
        #logits = self.class_output(penult_z).squeeze()
        return logits # logits.logsumexp(1)

def load_model(device,arch="resnet18",path=None):
    if path is not None:
        print("loading model from ",path)
        ckpt_dict = t.load(path)
        f = F(arch,n_classes)
        f.load_state_dict(ckpt_dict['model_state_dict'])
    else:
        f = F(arch,n_classes)
    f = f.to(device)
    return f

#dload_train,dload_test = load_data('/content/drive/MyDrive/CS480/Final/train')
dload_train,dload_test = load_data("train2",preloaded=False)
pdb.set_trace()
f = load_model(device,"resnet18")#,path=load_path)
params = f.model.parameters()
optim = t.optim.Adam(params,lr=0.0001,betas=[.9,.999])
n_epochs = 30

train_loss_history, train_acc_history = [],[]
test_loss_history, test_acc_history = [],[]

for epoch in range(n_epochs):
        
    losses,accs = [],[]
    for i,(x,y) in enumerate(dload_train):
        x,y = x.to(device),y.to(device)
        L = 0.

        logits = f(x)
        loss = nn.CrossEntropyLoss()(logits,y)
        acc = (logits.max(1)[1] == y).float().mean()
        losses.append(loss.item())
        accs.append(acc.item())
        L += loss

        optim.zero_grad()
        L.backward()
        optim.step()

    train_loss = np.mean(losses)
    train_acc = np.mean(accs)
    test_loss, test_acc = evaluate(f,dload_test,device)

    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)

    print("Epoch", epoch, "Train Loss: ", train_loss, "Train Accuracy: ", train_acc)
    print("Epoch", epoch, "Test Loss: ", test_loss, "Test Accuracy: ", test_acc)
    if epoch%5==0:
        plt.plot(train_loss_history,label="train loss")
        plt.plot(test_loss_history,label="test loss")
        plt.legend()
        plt.show()



fn = tr.Compose([tr.Pad(4,padding_mode='reflect'),tr.ToTensor()])