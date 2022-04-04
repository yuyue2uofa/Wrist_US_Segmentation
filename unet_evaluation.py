# -*- coding: utf-8 -*-

from skimage.io import imread
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv

import glob

import pickle
import re
import os

import segmentation_models_pytorch as smp

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# loss
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
    def forward(self, inputs, targets, smooth=1): 
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE 
        return Dice_BCE

# evaluation metrics
def evaluation_metrics(y_true, y_pred, smooth = 1):
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    precision = (intersection+smooth)/(np.sum(y_pred_f) + smooth)
    recall = (intersection+smooth)/(np.sum(y_true_f) + smooth)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    jaccard = (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth - intersection)
    return(precision, recall, dice, jaccard)

# model
model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    activation = "sigmoid"
)

class SegmentationDataSet(data.Dataset):
    def __init__(self, inputs: list, targets: list):
        self.inputs = inputs
        self.targets = targets
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]
        # Load input and target
        x, y = imread(input_ID,0), imread(target_ID,0)
        # Preprocessing
        x = (x - np.min(x)) / np.ptp(x)
        metaph = np.zeros(y.shape)
        metaph[y==1] = 1
        metaph[y==4] = 1
        epiph = np.zeros(y.shape)
        epiph[y==2] = 1
        carpal = np.zeros(y.shape)
        carpal[y==3] = 1
        y = np.stack((metaph, epiph, carpal))
        
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)
        return x, y, target_ID[-29:]

# Model evaluation
def eval_model(dataloader):
    eval_loss = 0
    precisions = []
    recalls = []
    dices = []
    jaccards = []
    model.eval()
    with torch.no_grad():
        for images, masks, image_filename in validation_dataloader:

            images = images.cuda()
            preds = model(images)
            preds = preds.cpu()
            loss = DiceBCELoss().forward(preds, masks)
            eval_loss =+ loss.item()
            preds[preds>=0.5] = 1 
            preds[preds<0.5] = 0

            precision, recall, dice, jaccard = evaluation_metrics(masks, preds)
            precisions.append(precision)
            recalls.append(recall)
            dices.append(dice)
            jaccards.append(jaccard)
            pred = preds[0,0,:,:]*255
            #if not os.path.isdir(save_dir+'result/'):
                #os.makedirs(save_dir+'result/')
            #cv2.imwrite(save_dir+'result/'+image_filename[0], pred.cpu().numpy())

    return eval_loss, sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(dices)/len(dices), sum(jaccards)/len(jaccards)

# Create validation dataset
data_dir = '/data/US_Wrist_Old/Test/' #change
input_dir = data_dir + 'Images/'
target_dir = data_dir + 'Masks/'

inputs = sorted(glob.glob(input_dir + '*.png'))
targets = sorted(glob.glob(target_dir + '*.png'))
test_dataset = SegmentationDataSet(inputs=inputs, targets=targets)
print('dataset created')
# Initialization
batch_size = 1

test_dataloader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle = True)

save_dir = './result/UNet/1/'

PATH = save_dir + 'UNet_ResNet34_best_model.pt' #change
if PATH != '':
    model.load_state_dict(torch.load(PATH))

model = model.cuda()

eval_loss, precision_eval, recall_eval, dice_eval, jaccard_eval = eval_model(validation_dataloader)
print('save_dir:', save_dir)
print("\t"+"eval loss: "+str(eval_loss)+
    "\t"+"eval precision: "+str(precision_eval)+
    "\t"+"eval rcall: "+str(recall_eval)+
    "\t"+"eval dice: "+str(dice_eval)+
    "\t"+"eval jaccard: "+str(jaccard_eval))

