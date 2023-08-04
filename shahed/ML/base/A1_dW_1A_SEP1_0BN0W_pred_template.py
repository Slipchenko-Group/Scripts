from make_data import  make_data
from main_functions import validate
import torch
import torch.nn.functional as F
import torchani
import torch.utils.tensorboard
import numpy as np
import os

# this script is just to load the trained model and run predictions on whatever dataset you want

species, cspecies, aevs_elecpots, energies, mol_names = make_data(label_col='interaction_energy', subtractE = 0, csv_file='dataset')

device = energies.device
np.random.seed(0)
p80 = int(energies.shape[0]*0.8)
total_pts = np.arange(0, int(energies.shape[0]), dtype='int32')
np.random.shuffle(total_pts)

np.random.seed(0)
p80 = int(energies.shape[0]*0.8)
total_pts = np.arange(0, int(energies.shape[0]), dtype='int32')
np.random.shuffle(total_pts)

train_cspecies = cspecies[total_pts[0:]]
train_species = species[total_pts[0:]]
train_aep = aevs_elecpots[total_pts[0:]]
train_energies = energies[total_pts[0:]].float()
train_names = [mol_names[i] for i in total_pts[0:]]

test_cspecies = cspecies[total_pts[p80:]]
test_species = species[total_pts[p80:]]
test_aep = aevs_elecpots[total_pts[p80:]]
test_energies = energies[total_pts[p80:]].float()
test_names = [mol_names[i] for i in total_pts[p80:]]

model_ANI1x = torchani.models.ANI1x(periodic_table_index=True).to(device)
model_ANI1x_0 = model_ANI1x[0]

max_epochs = 5000
early_stopping_learning_rate = 1.0E-5
wdecay1 = 0.0001
wdecay2 = 0.0001
lr_f = 0.5
name = "name of the best.pt file"
aev_dim = 385
set_0bias = 1
add_0w = 0
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

#####################################################################################################
max_atoms = 15

def write_out(name, true, preds, species, field, mol_names, max_atoms):
        out = open(f'{name}_results.csv', 'w+')
        out.write('true,pred,atomic_num,mol_num,atom_num,EP,mol_id\n')
        for i in range(preds.shape[0]):
                for ii in range(max_atoms):
                    spot = str(species[i][ii])
                    #spot = ','.join([str(f) for f in sfield[i][ii]])
                    out.write(f'{true[i].item()},{preds[i].item()},{spot},{i},{ii},{field[i][ii][-1]},{mol_names[i]}\n')
        out.close()

model = torchani.ANIModel([H_network, C_network, N_network, O_network])
model.load_state_dict(torch.load(f'{name}_best.pt'))
model.to(device)

_, train_preds = model((train_cspecies, train_aep))
train_preds = train_preds.cpu().detach().numpy()
#_, test_preds = model((test_cspecies, test_aep))
#test_preds = test_preds.cpu().detach().numpy()
write_out(f'train_{name}', train_energies, train_preds, train_species.cpu().detach().numpy(),train_aep.cpu().detach().numpy(), train_names, max_atoms)
#write_out(f'test_{name}', test_energies, test_preds, test_species.cpu().detach().numpy(), test_aep.cpu().detach().numpy(), test_names, max_atoms)

