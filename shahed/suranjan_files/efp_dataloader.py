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
from torch.utils.data import Dataset, DataLoader, random_split


# dataset class i made for this project to work with the dataloader - shahed





def convert(species):
    cspecies = []
    species_dict = {1 : 0, 6 : 1, 7 : 2, 8 : 3}
    for i in species:
        cspecies.append(species_dict[i])
    return np.array(cspecies)

def cspecies2atom(species):
    atoms = []
    cspecies_dict = {0 : 'H', 1 : 'C', 2 : 'N', 3 : 'O'}
    for i in species:
        if i == -1:
            continue
        else:
            atoms.append(cspecies_dict[i])
    return np.array(atoms)

def pad_tensor(arr,num_atoms,max_atoms,type_arr):
    arr = arr
    # if max_atoms == num_atoms:
    #    return arr
    padding = max_atoms - num_atoms
    # coords or forces
    if type_arr == 'c' or type_arr =='f':
        for i in range(padding):
            arr = np.append(arr,[[0,0,0]],0)
    # cspecies
    if type_arr == 's':
        for i in range(padding):
            arr =np.append(arr,[-1],-1)
    # elecpot
    if type_arr == 'e':
        for i in range(padding):
            arr = np.append(arr,[[0]],0)
    return arr


    

# create a custom dataset class for pytorch
class EFPANIdataset(Dataset):
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
        coordinates = np.transpose(np.array([molecule['coord_x'],molecule['coord_y'],molecule['coord_z']]))
        coordinates = pad_tensor(coordinates,num_atoms,self.max_atoms,'c')
        forces = np.transpose(np.array([molecule['force_x'],molecule['force_y'],molecule['force_z']]))
        forces = pad_tensor(forces,num_atoms,self.max_atoms,'f')
        energies = molecule[self.label].unique().item()
        cspecies = np.array(convert(molecule['atomic_number']))
        cspecies = pad_tensor(cspecies,num_atoms,self.max_atoms,'s')
        elecpot = np.expand_dims(np.array(molecule['elec_pot'].tolist()),-1)
        elecpot = pad_tensor(elecpot,num_atoms,self.max_atoms,'e')
        return {'cspecies': cspecies,'coordinates': coordinates,'energies': energies,'elecpot': elecpot,'forces': forces}


class EFPANIdataset2(Dataset):
    # define __init__ which just reads the csv and puts it into a df
    def __init__(self,dataset,label,aev_computer):
        self.df = pd.read_csv(dataset)
        self.label = label
        molecules = list(self.df['molecule'].unique())
        atoms_per_mol = self.df.groupby(['mol_id', 'molecule']).agg({'atom':'count'}).reset_index().drop('mol_id', axis=1).drop_duplicates()
        self.atoms_dict = atoms_per_mol.set_index('molecule').to_dict()['atom']
        self.max_atoms = max(self.atoms_dict.values())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.aev_computer = aev_computer

    #define __len__, which returns the total number of datapoints (molecules) in the dataset
    def __len__(self):
        return len(self.df['mol_id'].unique())
    
    # define __getitem__, which returns one datapoint (molecule) with all associated labels at idx
    def __getitem__(self,index):
        mol_name = self.df['mol_id'].unique()[index]
        molecule = self.df[self.df['mol_id'] == mol_name]
        num_atoms = len(self.df[self.df['mol_id']==mol_name])
        coordinates = np.transpose(np.array([molecule['coord_x'],molecule['coord_y'],molecule['coord_z']]))
        coordinates = pad_tensor(coordinates,num_atoms,self.max_atoms,'c')
        coordinates = torch.tensor(coordinates).to(self.device).requires_grad_(True)
        forces = np.transpose(np.array([molecule['force_x'],molecule['force_y'],molecule['force_z']]))
        forces = pad_tensor(forces,num_atoms,self.max_atoms,'f')
        forces = torch.tensor(forces).to(self.device)
        energies = molecule[self.label].unique().item()
        energies = torch.tensor(energies).to(self.device)
        cspecies = np.array(convert(molecule['atomic_number']))
        cspecies = pad_tensor(cspecies,num_atoms,self.max_atoms,'s')
        cspecies = torch.tensor(cspecies).to(self.device)
        elecpot = np.expand_dims(np.array(molecule['elec_pot'].tolist()),-1)
        elecpot = pad_tensor(elecpot,num_atoms,self.max_atoms,'e')
        elecpot = torch.tensor(elecpot).to(self.device)
        aevs = self.aev_computer((cspecies.unsqueeze(0),coordinates.unsqueeze(0)))[1].to(self.device).requires_grad_(True)
        aeps = torch.cat((aevs,elecpot.unsqueeze(0)),axis=2).to(self.device).requires_grad_(True).to(torch.float32).squeeze(0)
        return {'cspecies': cspecies,'coordinates': coordinates,'energies': energies,'elecpot': elecpot,'forces': forces,'aevs': aevs,'aeps':aeps}

class EFPANIdataset3(Dataset):
    # define __init__ which just reads the csv and puts it into a df
    def __init__(self,dataset,label,model_net):
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
        coordinates = np.transpose(np.array([molecule['coord_x'],molecule['coord_y'],molecule['coord_z']]))
        coordinates = pad_tensor(coordinates,num_atoms,self.max_atoms,'c')
        forces = np.transpose(np.array([molecule['force_x'],molecule['force_y'],molecule['force_z']]))
        forces = pad_tensor(forces,num_atoms,self.max_atoms,'f')
        energies = molecule[self.label].unique().item()
        cspecies = np.array(convert(molecule['atomic_number']))
        cspecies = pad_tensor(cspecies,num_atoms,self.max_atoms,'s')
        elecpot = np.expand_dims(np.array(molecule['elec_pot'].tolist()),-1)
        elecpot = pad_tensor(elecpot,num_atoms,self.max_atoms,'e')
        return {'cspecies': cspecies,'coordinates': coordinates,'energies': energies,'elecpot': elecpot,'forces': forces}


# create a custom dataset class for pytorch
# in this dataset class, i will prepare all the data in the __init__
class EFPANIdataset4(Dataset):
    # define __init__ which just reads the csv and puts it into a df
    # will also make it so that it prepares all the torch tensors here
    def __init__(self,dataset,label,aev_computer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_ANI1x = torchani.models.ANI1x(periodic_table_index=True).to(self.device)
        self.aev_computer = aev_computer
        self.all_coordinates = []
        self.all_cspecies = []
        self.all_elecpots = []
        self.all_energies = []
        self.all_forces = []
        self.all_atoms = []
        self.all_shift = []
        self.df = pd.read_csv(dataset)
        self.label = label
        molecules = list(self.df['molecule'].unique())
        mol_ids = list(self.df['mol_id'].unique())
        atoms_per_mol = self.df.groupby(['mol_id', 'molecule']).agg({'atom':'count'}).reset_index().drop('mol_id', axis=1).drop_duplicates()
        self.atoms_dict = atoms_per_mol.set_index('molecule').to_dict()['atom']
        self.max_atoms = max(self.atoms_dict.values())
        # add a for loop here
        # loop through the mol_ids and make the arrays and append them in
        for mol in mol_ids:
            mol_df = self.df[(self.df['mol_id']==mol)]
            num_atoms = len(mol_df)
            self.all_coordinates.append(pad_tensor((mol_df[['coord_x','coord_y','coord_z']].to_numpy()),num_atoms,self.max_atoms,'c'))
            self.all_cspecies.append(pad_tensor(convert(mol_df['atomic_number'].to_numpy()),num_atoms,self.max_atoms,'s'))
            self.all_energies.append(mol_df[self.label].unique().item())   
            # multiplying forces by 1.88973 to convert from Hartree/Bohr to Hartree/Angstrom
            self.all_forces.append(pad_tensor((mol_df[['force_x','force_y','force_z']].to_numpy()),num_atoms,self.max_atoms,'f')*1.88973)
            self.all_elecpots.append(pad_tensor((np.expand_dims(mol_df['elec_pot'].to_numpy(),-1)),num_atoms,self.max_atoms,'e'))
        for i in self.all_cspecies:
            self.all_atoms.append(cspecies2atom(i))
        for i in self.all_atoms:
            self.all_shift.append(sum([model_ANI1x.sae_dict[j] for j in i]))
    #define __len__, which returns the total number of datapoints (molecules) in the dataset
    def __len__(self):
        return len(self.df['mol_id'].unique())
    
    # define __getitem__, which returns one datapoint (molecule) with all associated labels at idx
    def __getitem__(self,index):
        # cspecies = torch.tensor(self.all_cspecies[index]).to(self.device)
        # coordinates = torch.tensor(self.all_coordinates[index]).to(self.device)
        # energies = self.all_energies[index]
        # elecpot = torch.tensor(self.all_elecpots[index]).to(self.device)
        # forces = torch.tensor(self.all_forces[index]).to(self.device)
        cspecies = self.all_cspecies[index]
        coordinates = self.all_coordinates[index]
        energies = self.all_energies[index]
        elecpot = self.all_elecpots[index]
        forces = self.all_forces[index]
        shift = self.all_shift[index]
        return {'cspecies': cspecies,'coordinates': coordinates,'energies': energies,'elecpot': elecpot,'forces': forces, 'shift': shift}




if __name__ == '__main__':
    dataset = '2vtl_test.csv'
    label = 'interaction_energy'
    df = pd.read_csv(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ANI1x = torchani.models.ANI1x(periodic_table_index=True).to(device)
    model_ANI1x_0 = model_ANI1x[0]
    aev_computer = model_ANI1x.aev_computer
    total_data = EFPANIdataset4(dataset,'interaction_energy',aev_computer)
    length = len(total_data)
    generator1 = torch.Generator().manual_seed(42)
    training, validation = random_split(total_data, [math.floor(length*0.8),math.ceil(length*0.2)],generator=generator1)
    dataloader1 = DataLoader(training,batch_size = 64,shuffle=True, num_workers=0)
    # dataloader2 = DataLoader(validation,shuffle=True,num_workers=6)
    # for idx, i in enumerate(dataloader):
    #     print(i)
    a = next(iter(dataloader1))
    # aevs = aev_computer((torch.tensor(a['cspecies']).to(device),torch.tensor(a['coordinates']).to(device)))[1]
    # aeps = torch.cat([aevs,torch.tensor(a['elecpot']).to(device)],axis=2).to(torch.float32)