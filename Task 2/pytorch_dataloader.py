import torch
import torch.nn as nn
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler

from torchvision import transforms

import numpy as np

class NumberDataLoader:
    def __init__(self):
        self.data_train = dset.SVHN('./data/', split='train',
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.43,0.44,0.47],
                                               std=[0.20,0.20,0.20])                           
                       ])
                      )
        
        self.data_test = dset.SVHN('./data/', split='test', 
                              transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.43,0.44,0.47],
                                                       std=[0.20,0.20,0.20])                           
                               ]))
        

        data_size = len(self.data_train)
        validation_split = .2
        split = int(np.floor(validation_split * data_size))
        indices = list(range(data_size))
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]
        self.val_indices = indices[:split]
        
    def get_data_loaders(self, batch_size: int=32, num_workers=0, _from=0, to=1000):
        data_size = len(self.data_train)
        validation_split = .2
        split = int(np.floor(validation_split * data_size))
        indices = list(range(data_size))
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]
        self.val_indices = val_indices
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = torch.utils.data.DataLoader(self.data_train, batch_size=batch_size, 
                                                   sampler=train_sampler, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(self.data_train, batch_size=batch_size,
                                             sampler=val_sampler, num_workers=num_workers)
        return train_loader, val_loader

    
    
    
    
    
    
    
    
    def get_train_subset(self, _form=0, to=1500, batch_size=32, num_workers=0, shuffle=True):
        data_size = len(self.data_train)
        validation_split = .2
        split = int(np.floor(validation_split * (to - _form)))
        indices = list(range(data_size))
        indices = indices[_form: to]
        
        if shuffle:
            np.random.shuffle(indices)
        
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_set = torch.utils.data.Subset(self.data_train, train_indices)
        val_set = torch.utils.data.Subset(self.data_train, val_indices)
        
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                                   shuffle=True, num_workers=num_workers)
        
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)
        
        return train_loader, val_loader
        
        
        
        
        
        
        
        
