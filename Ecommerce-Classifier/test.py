import csv 
import pdb
import os
from shutil import copyfile

def get_data(path):
    rows = [] 
    with open(path, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for i,row in enumerate(csvreader): 
            if i!=0: rows.append(row)
    return rows
rows = get_data("train.csv")
data = {r[0]:r[1] for r in rows}

train_imgs = list(data.keys())

root = "shuffled-images"
for f in os.listdir(root):
    image_id = f.strip(".jpg")
    if image_id in train_imgs:
        category = os.path.join("train",data[image_id])
        source = os.path.join(root,f)
        copyfile(source,category+"/"+f)
