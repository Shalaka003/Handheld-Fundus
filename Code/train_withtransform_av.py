#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:40:46 2022

@author: nitk
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import Concatenate
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
from data import ReadData
from model import build_unet
from loss import DiceLoss, DiceBCELoss, CELoss
from utils import seeding, create_dir, epoch_time



def train(model, loader, optimizer, loss_fn):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        # x = x.to(device, dtype=torch.float32)
        # y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        
        weights = np.array([1.08,31.11,26.28])  # weight for background, artery ,vein for
        # artery_vein_Unet /new_data
        
        loss = loss_fn(y_pred, y, weights = weights)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            # x = x.to(device, dtype=torch.float32)
            # y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            
            weights = np.array([1.08,31.11,26.28])
            loss = loss_fn(y_pred, y, weights = weights)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def load_model(chkpt_path, model, optimizer):
    
    checkpoint = torch.load(chkpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    valid_loss_list = checkpoint['valid_loss_values']
    train_loss_list = checkpoint['train_loss_values']
    
    return model, epoch, loss, optimizer, valid_loss_list, train_loss_list

def load_last_best_model (chkpt_path, model, optimizer):
    checkpoint = torch.load(chkpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, epoch, loss, optimizer 

if __name__ == "__main__":
    
    """ Seeding """
    seeding(42)

    """ Directories """
    # create_dir("files")
    
    """ Hyperparameters """
    
    batch_size = 2
    num_epochs = 2
    lr = 1e-4
    
    """ """
    
    checkpoint_path = "files/AV_RITELES_unet.pth"
    AV_colourmap = [ [0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]]
    AV_classes = ['background', 'artery', 'vein', 'crossing', 'uncertain']
   
    """ Load dataset """
    data_path = "../../ExpData/artery_vein/artery_vein_Unet/new_data/"
    train_x = sorted(glob(os.path.join(data_path, "train", "images", "*.png")))
    train_y = sorted(glob(os.path.join(data_path, "train", "masks", "*.png")))

    valid_x = sorted(glob(os.path.join(data_path, "valid", "images", "*.png")))
    valid_y = sorted(glob(os.path.join(data_path, "valid", "masks", "*.png")))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)


    train_transform = A.Compose(
        [
            
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit= 10.0, p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.RandomResizedCrop (height = 512, width = 512, scale=(0.08, 1.0), ratio=(1.0, 1.0), interpolation=1, always_apply=False, p=0.5)
            
        ]
        )
    
    # valid_transform = A.Compose(
    #     [
    #         #A.Resize(480, 512), 
    #        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            
    #     ]
    #     )
    
    valid_transform = None

    """ """
    # device = torch.device('cpu')   ## GTX 1060 6GB
    model = build_unet()
    # model = model.to(device)
    
    
    """Optimizer and Scheduler """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = CELoss()
    
    """Load the saved model """
    if os.path.exists(checkpoint_path):
        model, epoch_num, best_valid_loss, optimizer, valid_loss_list, train_loss_list = load_model(checkpoint_path, model, optimizer)
    else:
        best_valid_loss = float("inf")
        epoch_num = 0
        valid_loss_list = []
        train_loss_list = []
        
    """ Training the model """ 
    for epoch in range(num_epochs):
        start_time = time.time()
        
        
        """ Dataset Augment and loader """
        train_dataset = ReadData(train_x, train_y, transform = train_transform, label_values =AV_colourmap)
        valid_dataset = ReadData(valid_x, valid_y, transform = valid_transform, label_values =AV_colourmap)
        
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
        
        )

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        """ train and validate """
        train_loss = train(model, train_loader, optimizer, loss_fn)
        train_loss_list.append(train_loss)
        valid_loss = evaluate(model, valid_loader, loss_fn)
        valid_loss_list.append(valid_loss)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            
            if epoch == (num_epochs -1) :
                torch.save({
                    'epoch': epoch_num+epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss,   
                    'train_loss_values': train_loss_list,
                    'valid_loss_values': valid_loss_list         
                    }, checkpoint_path)
            
            else :
                torch.save({
                    'epoch': epoch_num+epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss,
                
                    }, checkpoint_path)
            
        else:
            
            model, epoch_n, last_best_loss, optimizer = load_last_best_model(checkpoint_path, model, optimizer)
            
            if epoch == (num_epochs -1) :
                 torch.save({
                    'epoch': epoch_num+epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss,   
                    'train_loss_values': train_loss_list,
                    'valid_loss_values': valid_loss_list         
                    }, checkpoint_path)
            
            
            else:
                torch.save ({
                    'epoch': epoch_num+epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': last_best_loss,
                    
                    }, checkpoint_path)
            
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch_num+epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)
        
        #if epoch == (num_epochs -1) :
         #   model, epoch_num, best_valid_loss, optimizer = load_last_best_model(checkpoint_path, model, optimizer)
            
    """ Plotting loss curves   """   
    plt.plot(np.array(valid_loss_list), 'r', label = "validation loss")
    plt.plot(np.array(train_loss_list), 'g', label = "training loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curve")
    plt.legend()
    #plt.show()
   
    plt.savefig('av.png')

