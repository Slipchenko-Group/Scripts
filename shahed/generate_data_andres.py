import pandas as pd
import numpy as np
import os

ligands = [fl for fl in os.listdir(os.getcwd()) if fl.startswith('snp')]
path = os.getcwd()
# below are orignal claudia numbers
# interaction energy = system energy - gas energy
# corr energy = interaction energy - mm energy
# system energy = energy of qm region in presence of mm region
# gas energy = energy of qm region gas phase

# below are the new numbers
# system energy = seems to be energy of the qm region in presence of mm region, but is essentially qm + qm/mm interaction
# gas energy = energy of qm region gas phase
# interaction energy = system energy - gas energy, so qm/mm interaction

d = {'step' : [], 'atom' : [], 'atomic_number' : [], 'molecule' : [], 'solv_molecule' : [], 'coord_x' : [], 'coord_y' : [], 'coord_z' : [], 'force_x' : [], 'force_y' : [], 'force_z' : [], 'elec_pot' : [], 
'mol_id' : [], 'atom_id' : [], 'system_energy' : [], 'gas_energy' : []}
atom_types = {'C':6,'H':1,'O':8,'N':7}


for step, s in enumerate(ligands):
    os.chdir(f'{path}/{s}')
    with open('gas_phase/gas.xyz','r', encoding='utf-8', errors='ignore') as xyz_fl:
        xyz_out_lns = xyz_fl.readlines()
        for iii, ii in enumerate(xyz_out_lns):
            if len(ii.split()) == 4:
                xyz_out_lns = xyz_out_lns[iii:]
                break
        for atomid, ln1 in enumerate(xyz_out_lns):
            # if len(ln1) == 3 or len(ln1) == 2:
            #    continue
            d['step'].append(step)
            d['atom'].append(ln1.split()[0][0])
            d['atomic_number'].append(atom_types[ln1.split()[0][0]])
            # d['molecule'].append(s)
            d['molecule'].append('p32')
            d['solv_molecule'].append('water_protein')
            # i dont like grabbing the xyz coords from the gas.xyz cause it truncates
            # instead i grabbed the xyz coords from the libefp.out because those have more decimals
            # d['coord_x'].append(float(ln1.split()[1]))
            # d['coord_y'].append(float(ln1.split()[2]))
            # d['coord_z'].append(float(ln1.split()[3]))
            d['mol_id'].append(f'p32_{s}')
            d['atom_id'].append(f'p32_{s}_{atomid}')
    with open('qm.log','r') as qmlog_fl:
        qmlog_ln = qmlog_fl.readlines()[0]
        for i in range(len(xyz_out_lns)):
             d['system_energy'].append(float(qmlog_ln.split()[-1]))
    with open('grad.xyz', 'r') as grad_fl:
        gradfl_lns = grad_fl.readlines()[2:]
        for ln2 in gradfl_lns:
            d['force_x'].append(float(ln2.split()[1]))
            d['force_y'].append(float(ln2.split()[2]))
            d['force_z'].append(float(ln2.split()[3]))
    with open('libefp.out','r') as libefp_fl:
        libefp_lns = libefp_fl.readlines()
        for idx, ln3 in enumerate(reversed(libefp_lns)):
            if 'ELECTROSTATIC POTENTIAL ON FRAGMENT 1' in ln3:
                efp_out = libefp_lns[-idx:-4]
                for e in efp_out:
                    d['elec_pot'].append(float(e.split()[-1]))
                    d['coord_x'].append(float(e.split()[1]))
                    d['coord_y'].append(float(e.split()[2]))
                    d['coord_z'].append(float(e.split()[3]))
    with open('gas_phase/sp.out') as gas_fl:
        gas_out_lns = gas_fl.readlines()
        for ln4 in reversed(gas_out_lns):
            if 'FINAL ENERGY:' in ln4:
                for i in range(len(xyz_out_lns)):
                    d['gas_energy'].append(float(ln4.split()[-2]))
    os.chdir('../')
df = pd.DataFrame(data=d)
df['interaction_energy'] = df['system_energy'] - df['gas_energy']
df.to_csv('andres_ligand_dataset_4bvn_p32.csv',index=False)
