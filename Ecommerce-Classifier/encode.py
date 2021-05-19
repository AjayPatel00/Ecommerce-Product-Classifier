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
n_classes = 27
genders_dict = {'Boys': 0, 'Girls': 1, 'Men': 2, 'Unisex': 3, 'Women': 4}
baseColours_dict = {'Beige': 0, 'Black': 1, 'Blue': 2, 'Bronze': 3, 'Brown': 4, 'Burgundy': 5, 'Charcoal': 6, 'Coffee Brown': 7, 'Copper': 8, 'Cream': 9, 'Fluorescent Green': 10, 'Gold': 11, 'Green': 12, 'Grey': 13, 'Grey Melange': 14, 'Khaki': 15, 'Lavender': 16, 'Lime Green': 17, 'Magenta': 18, 'Maroon': 19, 'Mauve': 20, 'Metallic': 21, 'Multi': 22, 'Mushroom Brown': 23, 'Mustard': 24, 'Navy Blue': 25, 'Nude': 26, 'Off White': 27, 'Olive': 28, 'Orange': 29, 'Peach': 30, 'Pink': 31, 'Purple': 32, 'Red': 33, 'Rose': 34, 'Rust': 35, 'Sea Green': 36, 'Silver': 37, 'Skin': 38, 'Steel': 39, 'Tan': 40, 'Taupe': 41, 'Teal': 42, 'Turquoise Blue': 43, 'White': 44, 'Yellow': 45}
seasons_dict = {'Fall': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
usages_dict = {'Casual': 0, 'Ethnic': 1, 'Formal': 2, 'Party': 3, 'Smart Casual': 4, 'Sports': 5, 'Travel': 6}
categories_dict = {'Accessories': 0, 'Apparel Set': 1, 'Bags': 2, 'Belts': 3, 'Bottomwear': 4, 'Cufflinks': 5, 'Dress': 6, 'Eyewear': 7, 'Flip Flops': 8, 'Fragrance': 9, 'Free Gifts': 10, 'Headwear': 11, 'Innerwear': 12, 'Jewellery': 13, 'Lips': 14, 'Loungewear and Nightwear': 15, 'Makeup': 16, 'Nails': 17, 'Sandal': 18, 'Saree': 19, 'Scarves': 20, 'Shoes': 21, 'Socks': 22, 'Ties': 23, 'Topwear': 24, 'Wallets': 25, 'Watches': 26}

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

def image_loader(image_name):
    fn = tr.Compose([tr.ToTensor()])
    image = Image.open(image_name)
    image = fn(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

def load_img_encoder():
    encoder = F('resnet18',n_classes)
    encoder.load_state_dict(t.load("imgf3.pt",map_location=t.device('cpu')))
    encoder.model.fc = nn.Linear(encoder.num_ftrs,512)
    encoder.eval()
    return encoder

encoder = load_img_encoder()
def encode_image(row):
    path = 'shuffled-images/'+str(row['id'])+'.jpg'
    im = image_loader(path)
    enc = encoder(im).detach().numpy()
    return enc[0]

def encode_features(row):
    g = np.zeros(len(genders_dict))
    b = np.zeros(len(baseColours_dict))
    s = np.zeros(len(seasons_dict))
    u = np.zeros(len(usages_dict))
    g[genders_dict[row['gender']]] = 1
    b[baseColours_dict[row['baseColour']]] = 1
    s[seasons_dict[row['season']]] = 1
    u[usages_dict[row['usage']]] = 1
    enc = np.concatenate((g,b,s,u))
    return enc

def get_data(path):
    file = open(path,'r')
    df = pd.read_csv(file)
    x = []
    for i, row in df.iterrows():
        enc1 = encode_image(row)
        enc2 = encode_features(row)
        enc = np.concatenate((enc1,enc2))
        x.append(enc)
        if i%1000==0: print(i)
    x = t.tensor(x)
    return x

x = get_data("test.csv")

pdb.set_trace()