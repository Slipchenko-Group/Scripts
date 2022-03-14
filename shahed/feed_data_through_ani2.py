import torch
import torchani
import numpy as np
import pandas as pd
import os


#this script is used to feed in molecular coordinates and atoms from Claudia's data generation format
#into whatever ANI model is necessary. This is only for seeing what vanilla ANI models predict.


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchani.models.ANI1x(periodic_table_index=True)[0].to(device)

data = [fl for fl in os.listdir(os.getcwd()) if fl.endswith('.csv')]
#atoms_idx_dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
data_dict = {'energy': [], 'mol_name': []}


for d in data:
    with open(d, 'r') as fl:
        out_lns = fl.readlines()
        out_lns = out_lns[1:]
        step = 0
        for idx, ln in enumerate(out_lns):
            if ln.split(',')[0] != step:
                step = ln.split(',')[0]
                mol_name = ln.split(',')[9]
                sp = []
                co = []
            co.append([float(x)for x in ln.split(',')[5:8]])
            sp.append(int(ln.split(',')[2]))
            if out_lns[-1] == out_lns[idx] or out_lns[idx].split(',')[0] != out_lns[idx+1].split(',')[0] and idx != 0:
            #if idx != 0 and out_lns[idx].split(',')[9] != out_lns[idx-1].split(',')[9] or out_lns[-1] ==out_lns[idx]:
                coords = torch.tensor([co], requires_grad=True, device=device)
                species = torch.tensor([sp],device=device)
                energy = model((species,coords)).energies
                for i in range(len(sp)):
                    data_dict['energy'].append(energy.item())
                    data_dict['mol_name'].append(mol_name)


df = pd.DataFrame.from_dict(data_dict)
df.to_csv(r'ANI_prediction_data.csv', index=False, header=True)