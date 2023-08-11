import torch
import torchani
import os
import math
import tqdm
import pandas as pd
import numpy as np
from torch import nn
from torch.nn.functional import pad
import torch.nn.functional as F
import torchani.utils
import gc
from collections import Counter
# helper function to convert energy unit from Hartree to kcal/mol
from torchani.units import hartree2kcalmol
from main_functions import *
from torch.utils.data import Dataset, DataLoader


# dataset class i made for this project to work with the dataloader

dataset = '2vtl_test.csv'
label = 'interaction_energy'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_ANI1x = torchani.models.ANI1x(periodic_table_index=True).to(device)
model_ANI1x_0 = model_ANI1x[0]
aev_computer = model_ANI1x.aev_computer

df = pd.read_csv(dataset)

def convert(species):
    cspecies = []
    species_dict = {1 : 0, 6 : 1, 7 : 2, 8 : 3}
    for i in species:
        cspecies.append(species_dict[i])
    return cspecies
def pad_tensor(arr,num_atoms,max_atoms,type_arr):
    arr = arr
    # if max_atoms == num_atoms:
    #    return arr
    padding = max_atoms - num_atoms
    # coords or forces
    if type_arr == 'c' or type_arr =='f':
        for i in range(padding):
            arr = np.append(arr,[[[0,0,0]]],1)
    # cspecies
    if type_arr == 's':
        for i in range(padding):
            arr =np.append(arr,[[-1]],-1)
    # elecpot
    if type_arr == 'e':
        for i in range(padding):
            arr = np.append(arr,[[[0]]],1)
    return arr


    

# create a custom dataset class for pytorch
class EFPANIdataset():
    # define __init__ which just reads the csv and puts it into a df
    def __init__(self,dataset,label):
        self.df = pd.read_csv(dataset)
        self.label = label
        molecules = list(self.df['molecule'].unique())
        atoms_per_mol = self.df.groupby(['mol_id', 'molecule']).agg({'atom':'count'}).reset_index().drop('mol_id', axis=1).drop_duplicates()
        self.atoms_dict = atoms_per_mol.set_index('molecule').to_dict()['atom']
        self.max_atoms = max(self.atoms_dict.values())
    
    #define __len__, which returns the total number of datapoints (molecules) in the dataset
    def __len__(self):
        return len(self.df['mol_id'].unique())
    
    # define __getitem__, which returns one datapoint (molecule) with all associated labels at idx
    def __getitem__(self,index):
        mol_name = self.df['mol_id'].unique()[index]
        molecule = self.df[self.df['mol_id'] == mol_name]
        num_atoms = len(self.df[self.df['mol_id']==mol_name])
        coordinates = np.expand_dims(np.transpose(np.array([molecule['coord_x'],molecule['coord_y'],molecule['coord_z']])),0)
        coordinates = pad_tensor(coordinates,num_atoms,self.max_atoms,'c')
        forces = np.expand_dims(np.transpose(np.array([molecule['force_x'],molecule['force_y'],molecule['force_z']])),0)
        forces = pad_tensor(forces,num_atoms,self.max_atoms,'f')
        energies = molecule[self.label].unique().item()
        cspecies = np.expand_dims(np.array(convert(molecule['atomic_number'])),0)
        cspecies = pad_tensor(cspecies,num_atoms,self.max_atoms,'s')
        elecpot = np.expand_dims(np.expand_dims(np.array(molecule['elec_pot'].tolist()),-1),0)
        elecpot = pad_tensor(elecpot,num_atoms,self.max_atoms,'e')
        return {'cspecies': cspecies,'coordinates': coordinates,'energies': energies,'elecpot': elecpot,'forces': forces}
    

if __name__ == '__main__':
    training = EFPANIdataset(dataset,'interaction_energy')
    dataloader = DataLoader(training,batch_size = 64,shuffle=True, num_workers=4)
    # for idx, i in enumerate(dataloader):
    #     print(i)
