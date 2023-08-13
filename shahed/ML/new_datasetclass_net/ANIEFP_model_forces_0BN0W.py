from make_data import  make_data
from main_functions import validate
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
# shut up pytorch 
import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device2 = 'cpu'

model_ANI1x = torchani.models.ANI1x(periodic_table_index=True).to(device)
model_ANI1x_0 = model_ANI1x[0]

dataset = '2vtl_test.csv'
label = 'interaction_energy'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
aev_computer = model_ANI1x.aev_computer


total_data = efp_dataloader.EFPANIdataset(dataset,'interaction_energy')
length = len(total_data)
generator1 = torch.Generator().manual_seed(42)
training, validation = random_split(total_data, [math.floor(length*0.8),math.ceil(length*0.2)],generator=generator1)
dataloader1 = DataLoader(training,batch_size = 2560,shuffle=True, num_workers=12, pin_memory=True)
dataloader2 = DataLoader(validation,batch_size = 2560, shuffle=True,num_workers=12,pin_memory=True)

max_epochs = 5000
early_stopping_learning_rate = 1.0E-5
#insert_wdecay1
#insert_wdecay2
#insert_lr_f
#insert_name
aev_dim = 385
set_0bias = 1
add_0w = 0
force_coefficient = 0.1  # controls the importance of energy loss vs force loss
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


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)

net = torchani.ANIModel([H_network, C_network, N_network, O_network])
net.apply(init_params)

def add_zeros(model, ref_model, atom, param, device=device):
    params = []
    atom_dict = {'H' : 0, 'C' : 1, 'N' : 2, 'O' : 3}
    for i in [0, 2, 4, 6]:
        if param == 'weight':
            diff_units = model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'].shape[1]-ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}'].shape[1]
        else:
            diff_units = model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'].shape[0]-ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}'].shape[0]
        ptensor = ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}']
        if param == 'weight' and i < 6 :
            ptensor = torch.cat((ptensor, torch.zeros(ptensor.shape[0], diff_units).to(device)), -1).to(device)
        params.append(ptensor)
    return(params)

def add_vals(model, ref_model, atom, param, device=device):
    params = []
    atom_dict = {'H' : 0, 'C' : 1, 'N' : 2, 'O' : 3}
    for i in [0, 2, 4, 6]:
        if param == 'weight':
            diff_units = model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'].shape[1]-ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}'].shape[1]
        else:
            diff_units = model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'].shape[0]-ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}'].shape[0]
        ptensor = ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}']
        if param == 'weight' and i < 6 :
            new_vals = model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'][0:ptensor.shape[0],ptensor.shape[1]:]
            ptensor = torch.cat((ptensor.to(device), new_vals.to(device)), -1).to(device)
        params.append(ptensor)
    return(params)

def set_params(model, ref_model, atom, param, setB0, add0):
    atom_dict = {'H' : 0, 'C' : 1, 'N' : 2, 'O' : 3}
    if add0 == 1:
        l0, l2, l4, l6 = add_zeros(model, ref_model, atom, param)
    else:
        l0, l2, l4, l6 = add_vals(model, ref_model, atom, param)
    layer_params = [l0, l2, l4, l6]
    layer_idx = [0, 2, 4]
    for ii in range(len(layer_idx)):
        i = layer_idx[ii]
        units_diff = model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'].shape[0]-ref_model.state_dict()[f'neural_networks.{atom}.{i}.{param}'].shape[0]
        if setB0 == 1 and param=='bias':
            model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'][...] = 0
        model.state_dict()[f'{atom_dict[atom]}.{i}.{param}'][0:-units_diff,...] = layer_params[ii]
    if param == 'weight':
        units_diff = model.state_dict()[f'{atom_dict[atom]}.6.{param}'].shape[1]-ref_model.state_dict()[f'neural_networks.{atom}.6.{param}'].shape[1]
        model.state_dict()[f'{atom_dict[atom]}.6.{param}'][...,0:-units_diff] = l6
    else:
        model.state_dict()[f'{atom_dict[atom]}.6.{param}'][:] = l6
        

set_params(net, model_ANI1x_0, 'H', 'bias', set_0bias, add_0w)
set_params(net, model_ANI1x_0, 'H', 'weight', 0, add_0w)
set_params(net, model_ANI1x_0, 'C', 'bias', set_0bias, add_0w)
set_params(net, model_ANI1x_0, 'C', 'weight', 0, add_0w)
set_params(net, model_ANI1x_0, 'N', 'bias', set_0bias, add_0w)
set_params(net, model_ANI1x_0, 'N', 'weight', 0, add_0w)
set_params(net, model_ANI1x_0, 'O', 'bias', set_0bias, add_0w)
set_params(net, model_ANI1x_0, 'O', 'weight', 0, add_0w)


model_net = torchani.nn.Sequential(net).to(device)
AdamW = torch.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight]},
    {'params': [H_network[2].weight], 'weight_decay': wdecay1},
    {'params': [H_network[4].weight], 'weight_decay': wdecay2},
    {'params': [H_network[6].weight]},
    # C networks
    {'params': [C_network[0].weight]},
    {'params': [C_network[2].weight], 'weight_decay': wdecay1},
    {'params': [C_network[4].weight], 'weight_decay': wdecay2},
    {'params': [C_network[6].weight]},
    # N networks
    {'params': [N_network[0].weight]},
    {'params': [N_network[2].weight], 'weight_decay': wdecay1},
    {'params': [N_network[4].weight], 'weight_decay': wdecay2},
    {'params': [N_network[6].weight]},
    # O networks
    {'params': [O_network[0].weight]},
    {'params': [O_network[2].weight], 'weight_decay': wdecay1},
    {'params': [O_network[4].weight], 'weight_decay': wdecay2},
    {'params': [O_network[6].weight]},
])
SGD = torch.optim.SGD([
    # H networks
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].bias]},
    {'params': [H_network[4].bias]},
    {'params': [H_network[6].bias]},
    # C networks
    {'params': [C_network[0].bias]},
    {'params': [C_network[2].bias]},
    {'params': [C_network[4].bias]},
    {'params': [C_network[6].bias]},
    # N networks
    {'params': [N_network[0].bias]},
    {'params': [N_network[2].bias]},
    {'params': [N_network[4].bias]},
    {'params': [N_network[6].bias]},
    # O networks
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].bias]},
    {'params': [O_network[4].bias]},
    {'params': [O_network[6].bias]},
], lr=1e-3)

AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=lr_f, patience=100, threshold=0)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=lr_f, patience=100, threshold=0)

latest_checkpoint = f'{name}_latest.pt'
if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    net.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])

tensorboard = torch.utils.tensorboard.SummaryWriter()


mse = torch.nn.MSELoss(reduction='none')
best_model_checkpoint = f'{name}_best.pt'
best_rmse = 10000000000
updated_lr = 0

def validate1():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    model_net.train(False)
    with torch.no_grad():
        for aa in dataloader1:
            coordinates = torch.tensor(aa['coordinates']).to(device).float().requires_grad_(True)
            cspecies = torch.tensor(aa['cspecies']).to(device)
            elecpots = torch.tensor(aa['elecpot']).to(device).float()
            true_energies = torch.tensor(aa['energies']).to(device).float()
            aevs = aev_computer((cspecies,coordinates))[1].to(device)
            aeps = torch.cat((aevs,elecpots),axis=2).to(device)
            _, predicted_energies = model_net((cspecies, aeps))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]
    model_net.train(True)
    return hartree2kcalmol(math.sqrt(total_mse / count))

def validate2():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    model_net.train(False)
    with torch.no_grad():
        for aa in dataloader2:
            coordinates = torch.tensor(aa['coordinates']).to(device).float().requires_grad_(True)
            cspecies = torch.tensor(aa['cspecies']).to(device)
            elecpots = torch.tensor(aa['elecpot']).to(device).float()
            true_energies = torch.tensor(aa['energies']).to(device).float()
            aevs = aev_computer((cspecies,coordinates))[1].to(device)
            aeps = torch.cat((aevs,elecpots),axis=2).to(device)
            _, predicted_energies = model_net((cspecies, aeps))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]
    model_net.train(True)
    return hartree2kcalmol(math.sqrt(total_mse / count))






for _ in range(max_epochs):
    # rmse = validate(model_net, train_cspecies, train_aep, train_energies)
    # rmse2 = validate(model_net, test_cspecies, test_aep, test_energies)
    rmse = validate1()
    rmse2 = validate2()
    print(f'{_},{rmse},{rmse2}')
    learning_rate = AdamW.param_groups[0]['lr']
    if learning_rate < early_stopping_learning_rate:
        break
    if rmse < 5 and updated_lr == 0:
        AdamW.param_groups[0]['lr'] = learning_rate/10
        updated_lr = 1
        learning_rate = AdamW.param_groups[0]['lr']

    # checkpoint
    if rmse < best_rmse and abs(rmse-rmse2)<0.5:
    #if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(net.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse)
    SGD_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    for a in dataloader1:
        coordinates = torch.tensor(a['coordinates']).to(device).float().requires_grad_(True)
        cspecies = torch.tensor(a['cspecies']).to(device)
        elecpots = torch.tensor(a['elecpot']).to(device).float()
        true_energies = torch.tensor(a['energies']).to(device).float()
        true_forces = torch.tensor(a['forces']).to(device).float()
        num_atoms = (cspecies >= 0).sum(dim=1, dtype=true_energies.dtype)
        aevs = aev_computer((cspecies,coordinates))[1].to(device).requires_grad_(True)
        aeps = torch.cat((aevs,elecpots),axis=2).to(device).requires_grad_(True)
        _, predicted_energies = model_net((cspecies,aeps))
        energy_loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
        forces = -torch.autograd.grad((predicted_energies).sum(),coordinates, create_graph=True,retain_graph=True)[0]
        force_loss = (mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
        loss = energy_loss + force_coefficient * force_loss
        # loss = energy_loss
        
        
        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()

        for child in model_net.children():
            for pname, param in child.named_parameters():
                if param.grad is None:
                    continue
                if 'weight' in pname and '6.weight' not in pname:
                    param.grad[0:-1,:] = 0
                elif 'bias' in pname and '6.bias' not in pname:
                    param.grad[0:-1] = 0
                elif '6.weight' in pname:
                    param.grad[...,0:-1] = 0
                elif '6.bias' in pname:
                    param.grad[:] = 0

        set_params(net, model_ANI1x_0, 'H', 'bias', set_0bias, add_0w)
        set_params(net, model_ANI1x_0, 'H', 'weight', 0, add_0w)
        set_params(net, model_ANI1x_0, 'C', 'bias', set_0bias, add_0w)
        set_params(net, model_ANI1x_0, 'C', 'weight', 0, add_0w)
        set_params(net, model_ANI1x_0, 'N', 'bias', set_0bias, add_0w)
        set_params(net, model_ANI1x_0, 'N', 'weight', 0, add_0w)
        set_params(net, model_ANI1x_0, 'O', 'bias', set_0bias, add_0w)
        set_params(net, model_ANI1x_0, 'O', 'weight', 0, add_0w)
        AdamW.step()
        SGD.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(true_energies))

    torch.save({
        'nn': net.state_dict(),
        'AdamW': AdamW.state_dict(),
        'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)
