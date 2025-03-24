'''
Created November 22, 2024
Training NN for estimating motion parameters
'''

import os
from pathlib import Path
import numpy as np
from time import time

import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import encode.encode_op as eop
import cnn.nn_models as nn


#%%-----------------------------------------------------------------------------
#-------------------------------HELPER FUNCTIONS--------------------------------
#%%-----------------------------------------------------------------------------

def folder_compile(folder, motion_lv_list, nsims_list):
   out_list = []
   for file in sorted(os.listdir(folder)):
      for motion_lv in motion_lv_list:
         for sim in nsims_list:
               temp = os.path.join(folder, file, motion_lv, 'sim{}'.format(sim+1))
               # temp = (folder, file, motion_lv, 'sim{}'.format(sim+1))
               out_list.append(temp)
   return out_list

def NaiveCoilCombo(s_corrupted, U_temp, C):
   # IFFT and naive coil combination for a segment of k-space
   m_corrupted_temp = eop._ifft(s_corrupted*U_temp, axes = (1,2,3))
   return torch.sum(torch.conj(C) * m_corrupted_temp, axis = 0)

def save_checkpoint(model, optimizer, save_path, epoch, loss_train, loss_val):
   torch.save({
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'epoch': epoch,
      'loss_train': loss_train,
      'loss_val': loss_val
   }, save_path)

def load_checkpoint(model, optimizer, load_path):
   checkpoint = torch.load(load_path)
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   epoch = checkpoint['epoch']
   return model, optimizer, epoch

#%%-----------------------------------------------------------------------------
#-------------------------------TRAINING FUNCTIONS--------------------------------
#%%-----------------------------------------------------------------------------

def train_epoch(model, optimizer, U, data_loader):
   for i, temp_path in enumerate(data_loader):
      t1 = time()
      print("Batch {}".format(i+1))
      optimizer.zero_grad() #zero grad for each batch
      loss = torch.zeros(1, device='cuda')
      loss.requires_grad = True
      for j, path in enumerate(temp_path):
         t2 = time()
         print("Sample {}".format(j+1))
         s_corrupted = torch.load(path + "/s_corrupted.pt")
         path_parent1 = str(Path(path).parents[1]) #get path 2 directories up
         C = torch.load(path_parent1 + "/C.pt")
         m_MOVE_store = [NaiveCoilCombo(s_corrupted,U[i],C) for i in range(len(U))]
         #
         output = model(m_MOVE_store)
         loss = loss + model.loss(output, m_MOVE_store) #must not be in-place operation
         t3 = time()
         print("Elapsed time per sample: {} sec".format(t3 - t2))
         #
      loss = loss / len(temp_path) #average loss over batch
      loss.backward()
      optimizer.step() #update step
      t4 = time()
      print("Elapsed time per batch: {} sec".format(t4 - t1))
   return loss

def val_epoch(model, U, data_loader):
   loss = torch.zeros(1, device='cuda')
   for i, temp_path in enumerate(data_loader):
      t1 = time()
      print("Batch {}".format(i+1))
      for j, path in enumerate(temp_path):
         t2 = time()
         print("Sample {}".format(j+1))
         s_corrupted = torch.load(path + "/s_corrupted.pt")
         path_parent1 = str(Path(path).parents[1]) #get path 2 directories up
         C = torch.load(path_parent1 + "/C.pt")
         m_MOVE_store = [NaiveCoilCombo(s_corrupted,U[i],C) for i in range(len(U))]
         #
         output = model(m_MOVE_store)
         loss = loss + model.loss(output, m_MOVE_store) #must not be in-place operation
         t3 = time()
         print("Elapsed time per sample: {} sec".format(t3 - t2))
         #
   loss = loss/ ((i+1)*(j+1)) #average loss over all batches
   t4 = time()
   print("Elapsed time per batch: {} sec".format(t4 - t1))
   return loss

#%%-----------------------------------------------------------------------------
#-----------------------------------LOAD DATA-----------------------------------
#%%-----------------------------------------------------------------------------

#Set paths
root = r'/home/nghiemb/GenPyMoCo'
dpath = r'/home/nghiemb/Data/CC'
dpath_train = os.path.join(root,'data','train')
dpath_test = os.path.join(root,'data','test')
dpath_val = os.path.join(root,'data','val')

spath = os.path.join(root,'cnn','weights')

motion_lv_list = ['moderate', 'large']
nsims_list = [0,1] #number of simulations per motion level in above list

train_list = folder_compile(dpath_train, motion_lv_list, nsims_list)
val_list = folder_compile(dpath_val, motion_lv_list, nsims_list)

# train_list = train_list[:8]
# val_list = val_list[:4]

training_loader = DataLoader(train_list, batch_size=4, \
                             shuffle=1, generator=torch.Generator(device='cpu'))
validation_loader = DataLoader(val_list, batch_size=1, \
                               shuffle=0, generator=torch.Generator(device='cpu'))

U = torch.load(os.path.join(root,'data', 'U.pt'))

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
nb_epochs = 50
lr = 1e-3

model = nn.Model_A()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(nb_epochs):
   print('Epoch {}'.format(epoch+1))
   #
   #Training
   model.train(True) #track gradients
   avg_loss_train = train_epoch(model, optimizer, U, training_loader)
   #
   #Validation
   model.eval() #set model to evaluation mode
   with torch.no_grad(): #disable gradients
      avg_loss_val = val_epoch(model, U, validation_loader)
   #
   spath_temp = spath + r'/epoch{}_{}.pt'.format(epoch+1, timestamp)
   save_checkpoint(model, optimizer, spath_temp, epoch+1, avg_loss_train, avg_loss_val)


'''
m_GT = torch.load(dpath_test + r'/Sub18/m_GT.pt')
C = torch.load(dpath_test + r'/Sub18/C.pt')

m_corrupted = torch.load(dpath_test + r'/Sub18/large/sim1/m_corrupted.pt')
s_corrupted = torch.load(dpath_test + r'/Sub18/large/sim1/s_corrupted.pt')
C = torch.load(dpath_test + r'/Sub18/C.pt')

from utils.visualize import plot_views, plot_Mtraj
plot_views(abs(m_GT).detach().cpu().numpy())

'''