from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
import math

from csv import DictReader

class MyDataset(Dataset):
    def __init__(self, version='it6',split='train', joint_transform=None, img_transform=None, url_csv_file=None, file_suffix=None):

        super().__init__()
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.split = split
        self.images = []
        self.targets = []        
        self.version = '_'+version if split == 'train' else ''
        
        # LOAD SPLIT CSV FILE
        
        self.root_dir = url_csv_file
        with open(self.root_dir + self.split + file_suffix + self.version+ '.csv') as f:
            csv_file = DictReader(f)
            for row in csv_file:
                self.images.append(row["image_urls"])
                self.targets.append(row["target_urls"])

        
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """
        image = Image.open(self.images[index]) #.convert('RGB')        
        target = Image.open(self.targets[index])
       
         # Convertir la PIL image a un tensor manualmente para que no haga la normalizacion   
        if self.joint_transform is not None:
            image, target = self.joint_transform(image,target)
            target = torch.from_numpy(np.array(target))
        if self.img_transform is not None:
            image = self.img_transform(image)
                
        return image, target