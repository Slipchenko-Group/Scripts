import torch
import torch.nn.functional as F
import torchani
import torch.utils.tensorboard
import numpy as np
import os
import pandas as pd
import tqdm
import efp_dataloader
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torchani.units import hartree2kcalmol
import aep_calculator
# shut up pytorch 
import warnings
warnings.filterwarnings("ignore")

# we have to set cpu or gpu here
# its fine to use cpu since we are not training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device2 = 'cpu'

# we dont really need this line, this was here just incase i wanted to mess around with ani1x 
model_ANI1x = torchani.models.ANI1x(periodic_table_index=True).to(device)
model_ANI1x_0 = model_ANI1x[0]

# here we will specify the dataset
# the dataset used to train was 2vtl_ligand_dataset_v1_no_ammonia.csv, but 2vtl_one.csv
# is the dataset that only has 1 molecule/datapoint in it, so i can load it and test it on the network
# dataset = '2vtl_ligand_dataset_v1_no_ammonia.csv'
dataset = '2vtl_one.csv'

# we train to system_energy, so the network predicts system energy
# label = 'interaction_energy'
label = 'system_energy'
# probably dont need these lines
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
aev_computer = model_ANI1x.aev_computer

# here i load the dataset using the dataloader class i made
total_data = efp_dataloader.EFPANIdataset4(dataset,'system_energy',aev_computer)
length = len(total_data)
generator1 = torch.Generator().manual_seed(42)
# split it into training and validation
# because the dataset only has one molecule, there is no actual splitting, one of these two will be empty
training, validation = random_split(total_data, [math.floor(length*0.8),math.ceil(length*0.2)],generator=generator1)
dataloader1 = DataLoader(validation,batch_size = 256,shuffle=True, num_workers=6, pin_memory=True)
dataloader2 = DataLoader(validation,batch_size = 256, shuffle=True,num_workers=6,pin_memory=True)


# these are the initial parameters for initializing the AEP_computer
Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
species_order = ['H', 'C', 'N', 'O']
num_species = len(species_order)

# creating the AEP_computer object from my modified ANI sourceode
aep_computer = aep_calculator.AEPComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)


# these were from training, dont need these lines
max_epochs = 5000
early_stopping_learning_rate = 1.0E-5
#insert_wdecay1
#insert_wdecay2
#insert_lr_f

# name of the network, so the .pt file but without _best.pt
name = 'ANIEFP_model_forces_0BN0W_v2_21'
aev_dim = 385
set_0bias = 1
add_0w = 0
# force_coefficient = 0.1  # controls the importance of energy loss vs force loss

# setting up the network in torch
H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 161),
    torch.nn.CELU(0.1),
    torch.nn.Linear(161, 129),
    torch.nn.CELU(0.1),
    torch.nn.Linear(129, 97),
    torch.nn.CELU(0.1),
    torch.nn.Linear(97, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 145),
    torch.nn.CELU(0.1),
    torch.nn.Linear(145, 113),
    torch.nn.CELU(0.1),
    torch.nn.Linear(113, 97),
    torch.nn.CELU(0.1),
    torch.nn.Linear(97, 1)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 129),
    torch.nn.CELU(0.1),
    torch.nn.Linear(129, 113),
    torch.nn.CELU(0.1),
    torch.nn.Linear(113, 97),
    torch.nn.CELU(0.1),
    torch.nn.Linear(97, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 129),
    torch.nn.CELU(0.1),
    torch.nn.Linear(129, 113),
    torch.nn.CELU(0.1),
    torch.nn.Linear(113, 97),
    torch.nn.CELU(0.1),
    torch.nn.Linear(97, 1)
)

# this part takes the networks we just set up and combines it all
# its ANI code 
model = torchani.ANIModel([H_network, C_network, N_network, O_network])
# this loads the .pt file
model.load_state_dict(torch.load(f'{name}_best.pt',map_location=torch.device('cpu')))
model.to(device)

# this creates a pipeline from the aep_computer we made to the model we just loaded
# so now the new model object is called model_net
# and when we pass the input to the model_net like this: model_net((cspecies,coordinates,elecpots))
# it will first go through the AEP_computer, then the output of that goes to the model, then the output of that is returned
model_net = torchani.nn.Sequential(aep_computer,model).to(device)

# species = validation[0]['cspecies']
# coordinates = validation[0]['coordinates']

# here we just set  up the input data from the dataset that we loaded, feel free to look at them
coordinates = torch.unsqueeze(torch.tensor(validation[0]['coordinates']).to(device).float().requires_grad_(True),0)
cspecies = torch.unsqueeze(torch.tensor(validation[0]['cspecies']).to(device),0)
elecpots = torch.unsqueeze(torch.tensor(validation[0]['elecpot']).to(device).float(),0)
true_energies = torch.tensor(validation[0]['energies']).to(device).float()
energy_shift = torch.tensor(validation[0]['shift']).to(device).float()

# here we pass the input of the molecule through the network to predict energies
# energies are in hartress, and the forces should be in Hartrees/Angstrom
_, predicted_energies = model_net((cspecies, coordinates, elecpots))
predicted_energies = torch.add(predicted_energies,energy_shift)


# here we do autograd to compute the forces
forces = torch.autograd.grad((predicted_energies).sum(),coordinates, create_graph=True,retain_graph=True)[0]

print(predicted_energies,forces)