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
d2 = {'step' : [], 'atom' : [], 'atomic_number' : [], 'molecule' : [], 'solv_molecule' : [], 'coord_x' : [], 'coord_y' : [], 'coord_z' : [], 'force_x' : [], 'force_y' : [], 'force_z' : [], 'elec_pot' : [], 
'mol_id' : [], 'atom_id' : [], 'system_energy' : [], 'gas_energy' : []}

for step, s in enumerate(ligands):
    os.chdir(f'{path}/{s}')
    with open('gas_phase/gas.xyz','r', encoding='utf-8', errors='ignore') as xyz_fl:
        xyz_out_lns = xyz_fl.readlines()
        for iii, ii in enumerate(xyz_out_lns):
            if len(ii.split()) == 4:
                xyz_out_lns = xyz_out_lns[iii:]
                break
        for atomid, ln1 in enumerate(xyz_out_lns):
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
# this part grabs the gas phase points
for step2, s2 in enumerate(ligands):
    os.chdir(f'{path}/{s2}')
    with open('gas_phase/gas_grad/gas.xyz','r', encoding='utf-8', errors='ignore') as xyz_fl2:
        xyz_out_lns2 = xyz_fl2.readlines()
        for iii2, ii2 in enumerate(xyz_out_lns2):
            if len(ii2.split()) == 4:
                xyz_out_lns2 = xyz_out_lns2[iii2:]
                break
        step3 = d['step'][-1] + 1
        for atomid2, ln12 in enumerate(xyz_out_lns2):
            # d['step'].append(step2)
            d['step'].append(step3)
            d['atom'].append(ln12.split()[0][0])
            d['atomic_number'].append(atom_types[ln12.split()[0][0]])
            # d['molecule'].append(s)
            d['molecule'].append('p32')
            d['solv_molecule'].append('none')
            d['mol_id'].append(f'p32_gas_{s2}')
            d['atom_id'].append(f'p32_gas_{s2}_{atomid2}')
    with open('gas_phase/sp.out', 'r') as gas_energy_fl:
        gas_energylns = gas_energy_fl.readlines()
        for ln5 in reversed(gas_energylns):
            if 'FINAL ENERGY:' in ln5:
                for i2 in range(len(xyz_out_lns2)):
                    d['system_energy'].append(float(ln5.split()[-2]))
                    d['gas_energy'].append(float(ln5.split()[-2]))
                    d['elec_pot'].append(0)
    with open('gas_phase/gas.g96','r') as gas_coords_fl:
        gas_coords = gas_coords_fl.readlines()
        for g, ln6 in enumerate(gas_coords):
            if 'POSITION' in ln6:
                gas_coords_start = g + 1
            if 'BOX' in ln6:
                gas_coords_end = g - 1
                break 
        gas_coords_only = gas_coords[gas_coords_start:gas_coords_end]
        for g2 in gas_coords_only:
            d['coord_x'].append(float(g2.split()[4]))
            d['coord_y'].append(float(g2.split()[5]))
            d['coord_z'].append(float(g2.split()[6]))
    with open('gas_phase/gas_grad/grad.out', 'r') as gas_grad_fl:
        gas_grad_lns = gas_grad_fl.readlines()
        for grad_idx, ln7 in enumerate(gas_grad_lns):
            if 'Gradient units are Hartree/Bohr' in ln7:
                grad_start = grad_idx+3
            if 'Net gradient:' in ln7:
                grad_end = grad_idx - 1
        grads = gas_grad_lns[grad_start:grad_end]
        for grad_xyz in grads:
            d['force_x'].append(float(grad_xyz.split()[0]))
            d['force_y'].append(float(grad_xyz.split()[1]))
            d['force_z'].append(float(grad_xyz.split()[2]))
    os.chdir('../')
df = pd.DataFrame(data=d)
df['interaction_energy'] = df['system_energy'] - df['gas_energy']
df.to_csv('andres_ligand_dataset_4bvn_p32.csv',index=False)
