import torch
import torchani
import numpy as np
import pandas as pd
import os

# this script feeds in the entire system with qm region and solvent all extracted from qmefp output files. it feeds it into ani and returns just
# the qm region energy

# this script is a fixed version, it uses the reoriented geometry rather than the initial geometry
# also is currently modified to work with gromacs style atom names and not qchem

# version 3
# this version requires the efp-efp .out files to be placed in the directory in their own ./efp folder
# the script will calculate the true energy by taking the qmefp energy + the efpefp energy 

# version 4
# this version fixes the species used to feed in
# when doing model(species,coords).energies to let the network calculate the energy
# you need to give in the actual atomic numbers, and the netowrk will convert it to their internal
# but i was preconverting it and therefore getting garbage numbers 
# when doing the forward_prop/ computing aevs manually, you need to give it in terms of internal
# so now its fixed to do both properlly, hopefully
# now there are two species arrays, species and species2
# species contains unconverted and species2 contains converted

data = [fl for fl in os.listdir(os.getcwd()) if fl.endswith('qmefp.in.out') or fl.endswith('qmefp.out')]

data_dict = {'predicted energy': [], 'true energy': [],'total system energy':[], 'qm energy':[], 'solvent/efp energy': [], 'predicted interaction energy': [], 'interaction2': [], 'mol_name': []}
atoms_idx = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
species_idx = {1:'H', 6:'C', 7:'N', 8:'O'}
atoms_ani_idx = {'H':1, 'C':6, 'N':7, 'O':8}
atoms_ani_idx2 = {'H':0, 'C':1, 'N':2, 'O':3}
atoms_idx_dict = {1 : 'H', 6 : 'C', 7 : 'N', 8 : 'O'}
atoms_idx_dict2 = {0 : 'H', 1 : 'C', 2 : 'N', 3 : 'O'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchani.models.ANI1x(periodic_table_index=True)[0].to(device)


def forward_prop(species, aev_inp, model0):
    atoms_idx_dict = {0 : 'H', 1 : 'C', 2 : 'N', 3 : 'O'}
    energies = []
    L4_activations = []
    celufn = torch.nn.CELU(alpha=0.1)
    for i in range(species[0].shape[0]):
        num = species[0][i]
        nnum = atoms_idx_dict2[num.item()]
        aev_ = aev_inp[i].reshape(-1, 1)
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

for name, d in enumerate(data):
    with open(d,'r') as fl:
        out_lns = fl.readlines()
        efp = []
        qm = []
        coords = []
        species = []
        species2 = []
        start_efp = 0
        end_efp = 0
        start_qm = 0 
        end_qm = 0
        # mol_name = data[name].replace('_qmefp.in.out', '')
        mol_name = data[name].split('.')[0].replace('_qmefp','')
        for idx, ln in enumerate(out_lns):
            # if 'GEOMETRY OF EFP SUBSYSTEM\n' in ln:
            if 'Standard Nuclear Orientation' in ln:
                # start_efp = idx + 3
                start_qm = idx + 3
            # if 'User input:\n' in ln:
            if 'Nuclear Repulsion Energy =' in ln:
                # end_efp = idx - 4
                # start_qm = idx + 4
                end_qm = idx - 2
            # if '$rem\n' in ln:
            if 'REORIENTED EFP GEOMETRY' in ln:
                # end_qm = idx - 2
                start_efp = idx + 2
                for idx2, aaa in enumerate(out_lns[start_efp:]):
                    if len(aaa.split()) == 0:
                        end_efp = start_efp + idx2
                        break
            if 'Total energy in the final basis set' in ln:
                true_energy = float(ln.split()[-1])
                os.chdir('./efp')
                with open(f'./{mol_name}_qmefp.in.out', 'r') as efpfl:
                    efp_out_lns = efpfl.readlines()
                    for ln2 in efp_out_lns:
                        if 'TOTAL EFP ENERGY' in ln2:
                            true_energy += float(ln2.split()[-1])
                            break
                os.chdir('..')
        efp = out_lns[start_efp:end_efp]
        qm = out_lns[start_qm:end_qm]
        for i in efp:
            species.append(i.split()[0][3:])
            species2.append(i.split()[0][3:])
            sub_coords = []
            for ii in i.split()[1:]:
                sub_coords.append(float(ii))
            # coords.append(float(i.split()[1:]))
            coords.append(sub_coords)
        for j in qm:
            species.append(j.split()[1])
            species2.append(j.split()[1])
            sub_coords = []
            for jj in j.split()[2:]:
                sub_coords.append(float(jj))
            # coords.append(j.split()[1:])
            coords.append(sub_coords)
        for idx, k in enumerate(species):
            species[idx] = atoms_ani_idx[k[0]]
            species2[idx] = atoms_ani_idx2[k[0]]
        fcoords = torch.tensor([coords], requires_grad = True, device = device)
        fspecies = torch.tensor([species], device = device)
        fspecies2 = torch.tensor([species2], device = device)
        efpchunk = len(efp)
        qmfspecies = fspecies2[0][efpchunk:].unsqueeze(0)
        qmaev = model.aev_computer(((fspecies2,fcoords)))[1][0][efpchunk:]
        energies = forward_prop(qmfspecies,qmaev,model)[0]
        # i dont think this part is necessary anymore after the species fix
        # jk this part might still be needed
        # because forward prop doesnt auto subtract
        # but doing model((species,coords)) does
        for s in qmfspecies[0]:
            subtract = model.sae_dict[atoms_idx_dict2[s.item()]]
            energies = energies + subtract
        data_dict['predicted energy'].append(energies)
        data_dict['mol_name'].append(mol_name)
        data_dict['true energy'].append(true_energy)

        # this part will be the calcualtion of total system - QM only - solvent only to get interaction via ANI
        total_energy = model((fspecies,fcoords)).energies.item()
        qm_energy = model((fspecies[0][efpchunk:].unsqueeze(0), fcoords[0][efpchunk:].unsqueeze(0))).energies.item()
        efp_energy = model((fspecies[0][:efpchunk].unsqueeze(0), fcoords[0][:efpchunk].unsqueeze(0))).energies.item()

        # i dont think this part is necesaary anymore after the species fix
        # for t in fspecies[0]:
        #     subtract = model.sae_dict[atoms_idx_dict[t.item()]]
        #     total_energy = total_energy + subtract
        # for q in fspecies[0][efpchunk:]:
        #     subtract = model.sae_dict[atoms_idx_dict[q.item()]]
        #     qm_energy = qm_energy + subtract
        # for e in fspecies[0][:efpchunk]:
        #     subtract = model.sae_dict[atoms_idx_dict[e.item()]]
        #     efp_energy = efp_energy + subtract
        interaction_energy = total_energy - qm_energy - efp_energy
        data_dict['predicted interaction energy'].append(interaction_energy)
        data_dict['qm energy'].append(qm_energy)
        data_dict['solvent/efp energy'].append(efp_energy)
        data_dict['total system energy'].append(total_energy)

        int2 = energies - qm_energy
        data_dict['interaction2'].append(int2)

df = pd.DataFrame.from_dict(data_dict)
df.to_csv(r'ANI_general_predictions_data_v4.csv', index = False, header = True)
