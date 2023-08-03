import pandas as pd
import numpy as np
import os


fragments = [fl for fl in os.listdir(os.getcwd()) if fl.startswith('frag')]
#ligands = [fl for fl in os.listdir(os.getcwd()) if fl.startswith('snp')]
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

# this is v2 of the code. it is not an upgrade but i had to change a few things because of the new file naming system
# and did not want to upload over the old one

d = {'step' : [], 'atom' : [], 'atomic_number' : [], 'molecule' : [], 'solv_molecule' : [], 'coord_x' : [], 'coord_y' : [], 'coord_z' : [], 'force_x' : [], 'force_y' : [], 'force_z' : [], 'elec_pot' : [], 
'mol_id' : [], 'atom_id' : [], 'system_energy' : [], 'gas_energy' : []}
atom_types = {'C':6,'H':1,'O':8,'N':7}
d2 = {'step' : [], 'atom' : [], 'atomic_number' : [], 'molecule' : [], 'solv_molecule' : [], 'coord_x' : [], 'coord_y' : [], 'coord_z' : [], 'force_x' : [], 'force_y' : [], 'force_z' : [], 'elec_pot' : [], 
'mol_id' : [], 'atom_id' : [], 'system_energy' : [], 'gas_energy' : []}

for f in fragments:
    os.chdir(f'{path}/{f}')
    fragname = f.split('_')[0]
    solv = f.split('_')[-1]
    ligands = [fl for fl in os.listdir(os.getcwd())]
    for step, s in enumerate(ligands):
        os.chdir(f'{path}/{f}/{s}')
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
                d['molecule'].append(f'{fragname}')
                d['solv_molecule'].append(f'{solv}')
                # i dont like grabbing the xyz coords from the gas.xyz cause it truncates
                # instead i grabbed the xyz coords from the libefp.out because those have more decimals
                # d['coord_x'].append(float(ln1.split()[1]))
                # d['coord_y'].append(float(ln1.split()[2]))
                # d['coord_z'].append(float(ln1.split()[3]))
                d['mol_id'].append(f'{fragname}_{s}')
                d['atom_id'].append(f'{fragname}_{s}_{atomid}')
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
            if solv == 'protein':
                for idx, ln3 in enumerate(reversed(libefp_lns)):
                    if 'ELECTROSTATIC POTENTIAL ON FRAGMENT 1' in ln3:
                        # print(ln3,idx)
                        # print(libefp_lns[idx])
                        efp_out = libefp_lns[-idx:-4]
                        # print(efp_out)
                        for e in efp_out:
                            d['elec_pot'].append(float(e.split()[-1]))
                            d['coord_x'].append(float(e.split()[1]))
                            d['coord_y'].append(float(e.split()[2]))
                            d['coord_z'].append(float(e.split()[3]))
                        break
            else:
                for idx, ln3 in enumerate(libefp_lns):
                    if 'ELECTROSTATIC POTENTIAL ON FRAGMENT 0' in ln3:
                        # print(ln3, idx)
                        efp_out = libefp_lns[idx+1:idx+len(xyz_out_lns)+1]
                        for e in efp_out:
                            d['elec_pot'].append(float(e.split()[-1]))
                            d['coord_x'].append(float(e.split()[1]))
                            d['coord_y'].append(float(e.split()[2]))
                            d['coord_z'].append(float(e.split()[3]))
                        break
        with open('gas_phase/sp.out') as gas_fl:
            gas_out_lns = gas_fl.readlines()
            for ln4 in reversed(gas_out_lns):
                if 'FINAL ENERGY:' in ln4:
                    for i in range(len(xyz_out_lns)):
                        d['gas_energy'].append(float(ln4.split()[-2]))

# this part handles gas phase
    for step2, s2 in enumerate(ligands):
        os.chdir(f'{path}/{f}/{s2}')
        with open('gas_phase/gas.xyz','r', encoding='utf-8', errors='ignore') as xyz_fl2:
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
                d['molecule'].append(f'{fragname}')
                d['solv_molecule'].append('none')
                d['mol_id'].append(f'{fragname}_gas_{s2}')
                d['atom_id'].append(f'{fragname}_{s2}_{atomid2}')
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
                d['coord_x'].append(float(g2.split()[4])*10)
                d['coord_y'].append(float(g2.split()[5])*10)
                d['coord_z'].append(float(g2.split()[6])*10)
        with open('gas_phase/grad.out', 'r') as gas_grad_fl:
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
    #os.chdir('../')
    os.chdir(f'{path}')


df = pd.DataFrame(data=d)
df['interaction_energy'] = df['system_energy'] - df['gas_energy']
df.to_csv('2vtl_ligand_dataset_v1.csv',index=False)


#ignore this lazy code
df1 = df.loc[df['molecule'] == 'frag1']
df1.to_csv('frag1.csv',index=False)

df2 = df.loc[df['molecule'] == 'frag2']
df2.to_csv('frag2.csv',index=False)

df3 = df.loc[df['molecule'] == 'frag3']
df3.to_csv('frag3.csv',index=False)

df4 = df.loc[df['molecule'] == 'frag4']
df4.to_csv('frag4.csv',index=False)

df5 = df.loc[df['molecule'] == 'frag5']
df5.to_csv('frag5.csv',index=False)

df6 = df.loc[df['molecule'] == 'frag6']
df6.to_csv('frag6.csv',index=False)

df7 = df.loc[df['molecule'] == 'fragfull']
df7.to_csv('fragfull.csv',index=False)

df8 = df1.loc[df1['solv_molecule'] == 'acetonitrile']
df8.to_csv('frag1_acetonitrile.csv',index=False)
df9 = df1.loc[df1['solv_molecule'] == 'ammonia']
df9.to_csv('frag1_ammonia.csv',index=False)
df10 = df1.loc[df1['solv_molecule'] == 'hexane']
df10.to_csv('frag1_hexane.csv',index=False)
df11 = df1.loc[df1['solv_molecule'] == 'water']
df11.to_csv('frag1_water.csv',index=False)
df12 = df1.loc[df1['solv_molecule'] == 'none']
df12.to_csv('frag1_gas.csv',index=False)

df13 = df2.loc[df2['solv_molecule'] == 'acetonitrile']
df13.to_csv('frag2_acetonitrile.csv',index=False)
df14 = df2.loc[df2['solv_molecule'] == 'ammonia']
df14.to_csv('frag2_ammonia.csv',index=False)
df15 = df2.loc[df2['solv_molecule'] == 'hexane']
df15.to_csv('frag2_hexane.csv',index=False)
df16 = df2.loc[df2['solv_molecule'] == 'water']
df16.to_csv('frag2_water.csv',index=False)
df17 = df2.loc[df2['solv_molecule'] == 'none']
df17.to_csv('frag2_gas.csv',index=False)

df18 = df3.loc[df3['solv_molecule'] == 'acetonitrile']
df18.to_csv('frag3_acetonitrile.csv',index=False)
df19 = df3.loc[df3['solv_molecule'] == 'ammonia']
df19.to_csv('frag3_ammonia.csv',index=False)
df20 = df3.loc[df3['solv_molecule'] == 'hexane']
df20.to_csv('frag3_hexane.csv',index=False)
df21 = df3.loc[df3['solv_molecule'] == 'water']
df21.to_csv('frag3_water.csv',index=False)
df22 = df3.loc[df3['solv_molecule'] == 'none']
df22.to_csv('frag3_gas.csv',index=False)

df23 = df4.loc[df4['solv_molecule'] == 'acetonitrile']
df23.to_csv('frag4_acetonitrile.csv',index=False)
df24 = df4.loc[df4['solv_molecule'] == 'ammonia']
df24.to_csv('frag4_ammonia.csv',index=False)
df25 = df4.loc[df4['solv_molecule'] == 'hexane']
df25.to_csv('frag4_hexane.csv',index=False)
df26 = df4.loc[df4['solv_molecule'] == 'water']
df26.to_csv('frag4_water.csv',index=False)
df27 = df4.loc[df4['solv_molecule'] == 'none']
df27.to_csv('frag4_gas.csv',index=False)

df28 = df5.loc[df5['solv_molecule'] == 'acetonitrile']
df28.to_csv('frag5_acetonitrile.csv',index=False)
df29 = df5.loc[df5['solv_molecule'] == 'ammonia']
df29.to_csv('frag5_ammonia.csv',index=False)
df30 = df5.loc[df5['solv_molecule'] == 'hexane']
df30.to_csv('frag5_hexane.csv',index=False)
df31 = df5.loc[df5['solv_molecule'] == 'water']
df31.to_csv('frag5_water.csv',index=False)
df32 = df5.loc[df5['solv_molecule'] == 'none']
df32.to_csv('frag5_gas.csv',index=False)

df33 = df6.loc[df6['solv_molecule'] == 'acetonitrile']
df33.to_csv('frag6_acetonitrile.csv',index=False)
df34 = df6.loc[df6['solv_molecule'] == 'ammonia']
df34.to_csv('frag6_ammonia.csv',index=False)
df35 = df6.loc[df6['solv_molecule'] == 'hexane']
df35.to_csv('frag6_hexane.csv',index=False)
df36 = df6.loc[df6['solv_molecule'] == 'water']
df36.to_csv('frag6_water.csv',index=False)
df37 = df6.loc[df6['solv_molecule'] == 'none']
df37.to_csv('frag6_gas.csv',index=False)

df38 = df7.loc[df7['solv_molecule'] == 'acetonitrile']
df38.to_csv('fragfull_acetonitrile.csv',index=False)
df39 = df7.loc[df7['solv_molecule'] == 'ammonia']
df39.to_csv('fragfull_ammonia.csv',index=False)
df40 = df7.loc[df7['solv_molecule'] == 'hexane']
df40.to_csv('fragfull_hexane.csv',index=False)
df41 = df7.loc[df7['solv_molecule'] == 'water']
df41.to_csv('fragfull_water.csv',index=False)
df42 = df7.loc[df7['solv_molecule'] == 'none']
df42.to_csv('fragfull_gas.csv',index=False)
df43 = df7.loc[df7['solv_molecule'] == 'protein']
df43.to_csv('fragfull_protein.csv',index=False)


df44 = df.loc[df['solv_molecule'] != 'ammonia']
df44.to_csv('2vtl_ligand_dataset_v1_no_ammonia.csv',index=False)
