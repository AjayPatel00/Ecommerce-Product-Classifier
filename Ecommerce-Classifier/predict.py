import torch as t, torch.nn as nn, torch.nn.functional as tnnF
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision as tv, torchvision.transforms as tr
from torchvision import models
from collections import Counter
import matplotlib.pyplot as plt
from torch.autograd import Variable
from PIL import Image
import pdb
import csv
import pandas as pd
import os
import sys
import numpy as np
genders_dict = {'Boys': 0, 'Girls': 1, 'Men': 2, 'Unisex': 3, 'Women': 4}
baseColours_dict = {'Beige': 0, 'Black': 1, 'Blue': 2, 'Bronze': 3, 'Brown': 4, 'Burgundy': 5, 'Charcoal': 6, 'Coffee Brown': 7, 'Copper': 8, 'Cream': 9, 'Fluorescent Green': 10, 'Gold': 11, 'Green': 12, 'Grey': 13, 'Grey Melange': 14, 'Khaki': 15, 'Lavender': 16, 'Lime Green': 17, 'Magenta': 18, 'Maroon': 19, 'Mauve': 20, 'Metallic': 21, 'Multi': 22, 'Mushroom Brown': 23, 'Mustard': 24, 'Navy Blue': 25, 'Nude': 26, 'Off White': 27, 'Olive': 28, 'Orange': 29, 'Peach': 30, 'Pink': 31, 'Purple': 32, 'Red': 33, 'Rose': 34, 'Rust': 35, 'Sea Green': 36, 'Silver': 37, 'Skin': 38, 'Steel': 39, 'Tan': 40, 'Taupe': 41, 'Teal': 42, 'Turquoise Blue': 43, 'White': 44, 'Yellow': 45}
seasons_dict = {'Fall': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
usages_dict = {'Casual': 0, 'Ethnic': 1, 'Formal': 2, 'Party': 3, 'Smart Casual': 4, 'Sports': 5, 'Travel': 6}
categories_dict = {'Accessories': 0, 'Apparel Set': 1, 'Bags': 2, 'Belts': 3, 'Bottomwear': 4, 'Cufflinks': 5, 'Dress': 6, 'Eyewear': 7, 'Flip Flops': 8, 'Fragrance': 9, 'Free Gifts': 10, 'Headwear': 11, 'Innerwear': 12, 'Jewellery': 13, 'Lips': 14, 'Loungewear and Nightwear': 15, 'Makeup': 16, 'Nails': 17, 'Sandal': 18, 'Saree': 19, 'Scarves': 20, 'Shoes': 21, 'Socks': 22, 'Ties': 23, 'Topwear': 24, 'Wallets': 25, 'Watches': 26}
n_classes = 27
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                            #####                         
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]   ########                                
    return weight   

def evaluate(f,data):
    f.eval()
    losses,accs = [],[]
    for x,y in data:
        #x,y = x.to(device),y.to(device)
        logits = f(x)
        loss = nn.CrossEntropyLoss()(logits,y)
        acc = (logits.max(1)[1] == y).float().mean()
        losses.append(loss.item())
        accs.append(acc.item())
    f.train()
    return np.mean(losses),np.mean(accs)

def load_data(path_train,path_test,preloaded=True):
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
        dset = tv.datasets.ImageFolder(root=path_train,transform=fn)
        lbls = [dset[i][1].item() for i in range(len(dset))]
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
    else:
        loaded = np.load(path_train,allow_pickle=True)
        x = [_[0] for _ in loaded]
        y = [_[1] for _ in loaded]
        x,y = t.tensor(x), t.tensor(y,dtype=t.long)
        dset_train = TensorDataset(x,y)
        loaded = np.load(path_test,allow_pickle=True)
        x = [_[0] for _ in loaded]
        y = [_[1] for _ in loaded]
        x,y = t.tensor(x), t.tensor(y,dtype=t.long)
        dset_test = TensorDataset(x,y)

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

def load_model(arch="resnet18",path=None):
    if path is not None:
        print("loading model from ",path)
        ckpt_dict = t.load(path)
        f = F(arch,n_classes)
        f.load_state_dict(ckpt_dict['model_state_dict'])
    else:
        f = F(arch,n_classes)
    #f = f.to(device)
    return f

fn = tr.Compose([tr.ToTensor()])
Categories = ['Accessories', 'Apparel Set', 'Bags', 'Belts', 'Bottomwear', 'Cufflinks', 'Dress', 'Eyewear', 'Flip Flops', 'Fragrance', 'Free Gifts', 'Headwear', 'Innerwear', 'Jewellery', 'Lips', 'Loungewear and Nightwear', 'Makeup', 'Nails', 'Sandal', 'Saree', 'Scarves', 'Shoes', 'Socks', 'Ties', 'Topwear', 'Wallets', 'Watches']
def image_loader(image_name):
    image = Image.open(image_name)
    image = fn(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image#.cuda()  #assumes that you're using GPU

def predict(f,path):
    im = image_loader(path)
    #im.to(device)
    logits = f(im)
    category = Categories[logits.max(1)[1][0].item()]
    return category

def get_preds(f,path):
    f.eval()
    preds = []
    for i,im in enumerate(os.listdir(path)):
        im_path = os.path.join(path,im)
        id = im.strip(".jpg")
        category = predict(f,im_path)
        preds.append([id,category])
        if i%1000==0: print(i)
    f.train()
    return preds

f = F()
f.load_state_dict(t.load("imgf3.pt", map_location=t.device('cpu')))
f.eval()

preds = get_preds(f,"test_shuffled")

with open("model1.csv","w") as csvFile:
    print("id,category", file=csvFile)
    for id,cat in preds:
        print(id+","+cat,file=csvFile)

pdb.set_trace()