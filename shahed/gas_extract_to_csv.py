import os
import numpy as np
import pandas as pd

#this script is written to go through the gas phase qmefp calculations of qchem, take the geometry and system energy
#and put them into a csv to use for training with Claudias code
#it also skips all duplicates

outs = [fl for fl in os.listdir(os.getcwd()) if fl.endswith('gas.in.out')]
qmefp_dict = {'step' : [],'atom' : [],'atomic_num' : [], 'qm_molecule' : [], 'solv_molecule' : [], 'coord_x' : [], 'coord_y' : [], 'coord_z' : [], 'elec_pot' : [],
              'mol_id' : [], 'atom_id' : [],'system_energy' : [], 'gas_energy' : [], 'efp_energy' : []}
atom_dict = {'H' : 1, 'C' : 6, 'N' : 7, 'O' : 8}
qm_mol = 'methanol'
efp_energy = 0
for s, out in enumerate(outs):
    with open(out, 'r') as fl:
        geo_start = 0
        geo_end = 0
        energy = 0
        elec = 0
        solv_mol = out.split('_')[1]
        out_lns = fl.readlines()
        for idx, ln in enumerate(out_lns):
            if '$molecule' in ln:
                geo_start = idx + 2
            if '$end' in ln and len(out_lns[idx-1].split()) > 2:
                geo_end = idx
            if 'Total energy in the final basis set' in ln:
                energy = float(ln.split('=')[-1].replace('\n', ''))
        if outs[s].split('_')[2] == outs[s - 1].split('_')[2]:
            continue
        geom = out_lns[geo_start:geo_end]
        for a, i in enumerate(geom):
            qmefp_dict['step'].append(s + 1)
            qmefp_dict['atom'].append(i.split()[0])
            qmefp_dict['qm_molecule'].append(qm_mol)
            qmefp_dict['solv_molecule'].append(solv_mol)
            qmefp_dict['system_energy'].append(energy)
            qmefp_dict['gas_energy'].append(energy)
            qmefp_dict['efp_energy'].append(efp_energy)
            qmefp_dict['mol_id'].append(out.split('.')[0])
            qmefp_dict['atom_id'].append(out.split('.')[0]+str(a))
            qmefp_dict['coord_x'].append(i.split()[1])
            qmefp_dict['coord_y'].append(i.split()[2])
            qmefp_dict['coord_z'].append(i.split()[3])
            qmefp_dict['elec_pot'].append(elec)
            qmefp_dict['atomic_num'].append(atom_dict[i.split()[0]])
        qmefp_df = pd.DataFrame(qmefp_dict)
        qmefp_df['interaction_energy'] = qmefp_df['system_energy'] - qmefp_df['gas_energy']
        qmefp_df['system_energy_w/oEFP'] = qmefp_df['system_energy'] - qmefp_df['efp_energy']
        qmefp_df['corr_energy'] = qmefp_df['system_energy_w/oEFP'] - qmefp_df['gas_energy']


path = os.getcwd()
data = qmefp_df.rename(columns={'qm_molecule': 'molecule'})
data.to_csv(f'{path}/qmefp_libefp_gas_only_631Gd.csv', index=False)