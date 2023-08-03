import numpy as np
import torch
from torch.nn.functional import pad
from torchani.units import hartree2kcalmol
import math

def pd2arr(df, mol_name, device, label_col, n_steps, n_atoms):
    #creates a df for a specific molecule name in the original given df
    #so for us we might give it like mol1_centered_solv_53 or something
    #then it grabs the chunk of the df for that specified molecule and stores it
    mol_df = df[df['molecule']==mol_name]
    #empty arrays created for later
    coord_arrs = []
    elecpot_arrs = []
    energy_arrs = []
    molid_arrs = [] 
    for a in range(0, n_atoms):
        #grabs the column in the df of whatever your label_col is and puts it in
        #an np array
        atom_energy_arr = mol_df[[label_col]].to_numpy()[a::n_atoms]
        #grabs the mol id of the molecule that youre looking at
        #so like mol1_53_s_water_5 etc.
        #stores only 1 of it in an array
        atom_molid_arr = mol_df['mol_id'].values[a::n_atoms]
        #if a is 0 then we grab the energy and the molid and put it into the arrays
        #that we created before the loop
        #i guess this is so that we guarantee the first value and not have to worry
        #so if its CH4, the first one will be C in the loop,
        #and we just grab the values from that
        if a == 0:
            energy_arrs.append(atom_energy_arr)
            molid_arrs.extend(list(atom_molid_arr))
        atom_elecpot_arr = mol_df['elec_pot'].to_numpy()[a::n_atoms].reshape(n_steps, 1)
        atom_elecpot_arr = np.expand_dims(atom_elecpot_arr, axis=0).reshape(n_steps, 1, 1)
        elecpot_arrs.append(atom_elecpot_arr)
        atom_coord_arr = mol_df[['coord_x', 'coord_y', 'coord_z']].to_numpy()[a::n_atoms]
        atom_coord_arr = np.expand_dims(atom_coord_arr, axis=0).reshape(n_steps, 1, 3)
        coord_arrs.append(atom_coord_arr)

    coord_tensor = torch.tensor(np.concatenate(coord_arrs, axis=1))
    energy_tensor = torch.tensor(np.concatenate(energy_arrs, axis=1))
    elecpot_tensor = torch.tensor(np.concatenate(elecpot_arrs, axis=1))
    return(coord_tensor, energy_tensor, elecpot_tensor, molid_arrs)

#pads the input tensor
def pad_mol(t, mol_name, val, n_atoms, max_atoms):
    p = max_atoms - n_atoms
    if t.dim()==3:
        #pad 2nd dimension of tensor by 0,p
        #else pad 1st dimension by 0,p instead
        t_pad = pad(t, pad=(0,0,0,p,0,0), mode='constant', value=val)
    else:
        t_pad = pad(t, pad=(0,p), mode='constant', value=val)
    return(t_pad)

#input takes a numpy array
def convert_species(nums):
    species_dict = {1 : 0, 6 : 1, 7 : 2, 8 : 3}
    mol_nums = nums.copy()
    #goes through mol_nums and converts each one to the appropriate 
    #number found in species dict
    #i think it goes by atomic number
    #so 1 = H, 6 = C, 7 = N, 8 = O
    #and then it orders it by 1,2,3
    #i think this is because when training the atoms have to be in periodic order
    for n in range(mol_nums.shape[0]):
         mol_nums[n] = species_dict[mol_nums[n]]
    return(mol_nums)

#does the opposite of the previous function
def reconvert_species(nums):
    species_dict = {0 : 1, 1 : 6, 2 : 7, 3 : 8}
    mol_nums = nums.copy()
    #converts the 0,1,2,3 from the previous function back into atomic number
    for n in range(mol_nums.shape[0]):
         mol_nums[n] = species_dict[mol_nums[n]]
    return(mol_nums)


def forward_prop(species, aev_inp, model0):
    atoms_idx_dict = {0 : 'H', 1 : 'C', 2 : 'N', 3 : 'O'}
    energies = []
    L4_activations = []
    celufn = torch.nn.CELU(alpha=0.1)
    for i in range(species[0].shape[0]):
        num = species[0][i]
        nnum = atoms_idx_dict[num.item()]
        aev_ = aev_inp[0, i].reshape(-1, 1)
        A0 = torch.sum(torch.matmul(model0.state_dict()[f'neural_networks.{nnum}.0.weight'], aev_), -1)
        A0 = torch.add(A0, model0.state_dict()[f'neural_networks.{nnum}.0.bias'])
        Z0 = celufn(A0).reshape(-1, 1)
        A2 = torch.sum(torch.matmul(model0.state_dict()[f'neural_networks.{nnum}.2.weight'], Z0), -1)
        A2 = torch.add(A2, model0.state_dict()[f'neural_networks.{nnum}.2.bias'])
        Z2 = celufn(A2).reshape(-1, 1)
        A4 = torch.sum(torch.matmul(model0.state_dict()[f'neural_networks.{nnum}.4.weight'], Z2), -1)
        A4 = torch.add(A4, model0.state_dict()[f'neural_networks.{nnum}.4.bias'])
        Z4 = celufn(A4).reshape(-1, 1)
        A6 = torch.sum(torch.matmul(model0.state_dict()[f'neural_networks.{nnum}.6.weight'], Z4), -1)
        A6 = torch.add(A6, model0.state_dict()[f'neural_networks.{nnum}.6.bias'])
        energies.append(A6)
        L4_activations.append(A4.reshape(1, -1))
    A6_energy = sum(energies).item()
    A4_output = torch.cat(L4_activations, 0)
    return(A6_energy, A4_output)



def validate(model, features_species, features_aevs, labels):
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    true_energies = labels
    _, predicted_energies = model((features_species, features_aevs))
    total_mse = mse_sum(predicted_energies, true_energies).item()
    count = predicted_energies.shape[0]
    return hartree2kcalmol(math.sqrt(total_mse / count))


def validate2(model, features, labels):
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    true_energies = labels
    predicted_energies = model(features)
    total_mse = mse_sum(predicted_energies, true_energies).item()
    count = predicted_energies.shape[0]
    return hartree2kcalmol(math.sqrt(total_mse / count))
    #return (math.sqrt(total_mse / count))

