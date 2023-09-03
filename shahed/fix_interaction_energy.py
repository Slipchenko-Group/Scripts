import os
import pandas as pd


data = [fl for fl in os.listdir() if fl.endswith('.csv')]
folders = {'ammonia':'Ammonia','methane':'Methane','methanol':'Methanol','water':'Water','mobley':'MobleySolv'}
df = pd.read_csv(data[0])
path = os.getcwd()
systems = df['mol_id'].unique()


# i = systems[0]
# j = df[df['mol_id'] == i]

libefp_dispersion = []
libefp_exchangerep = []
qmefp_dispersion = []
qmefp_exchangerep = []
disp_exch_interaction = []

lib_disp_values = {}
lib_ex_values = {}
qm_disp_values = {}
qm_ex_values = {}
total_values = {}
for i in df.iterrows():
    solute = i[1][9].split('_')[0]
    mol_id = i[1][9]

    if i[1][9].split('_')[-1] == '0EP':
        number = 0
        disp_exch_interaction.append(number)
        libefp_dispersion.append(number)
        libefp_exchangerep.append(number)
        qmefp_dispersion.append(number)
        qmefp_exchangerep.append(number)
    else:
        if mol_id in lib_disp_values:
            libefp_dispersion.append(lib_disp_values[mol_id])
            libefp_exchangerep.append(lib_ex_values[mol_id])
            qmefp_dispersion.append(qm_disp_values[mol_id])
            qmefp_exchangerep.append(qm_ex_values[mol_id])
            disp_exch_interaction.append(total_values[mol_id])
        else:
            libefp_disp = 0
            libefp_exrp = 0
            qmefp_disp = 0
            qmefp_exrp = 0
            total_inter = 0 
            folder = folders[i[1][9].split('_')[0]]
            if solute == 'mobley':
                with open(f'{path}/G_LibEFP/ElecPot/{folder}/{mol_id}.in.out') as fl:
                    lns = fl.readlines()
                    for j1 in lns:
                        if 'DISPERSION ENERGY' in j1:
                            libefp_disp = float(j1.split()[-1]) 
                            lib_disp_values[f'{mol_id}'] = libefp_disp
                            libefp_dispersion.append(libefp_disp)
                        if 'EXCHANGE REPULSION ENERGY' in j1:
                            libefp_exrp = float(j1.split()[-1])
                            lib_ex_values[f'{mol_id}'] = libefp_exrp
                            libefp_exchangerep.append(libefp_exrp)
                            break
                with open(f'{path}/H_QMEFP/{folder}/{mol_id}_qmefp.in.out') as fl2:
                    lns2 = fl2.readlines()
                    for j2 in lns2:
                        if 'DISPERSION ENERGY' in j2:
                            qmefp_disp = float(j2.split()[-1])
                            qm_disp_values[f'{mol_id}'] = qmefp_disp
                            qmefp_dispersion.append(qmefp_disp)
                        if 'EXCHANGE-REPULSION ENERGY' in j2:
                            qmefp_exrp = float(j2.split()[-1])
                            qm_ex_values[f'{mol_id}'] = qmefp_exrp
                            qmefp_exchangerep.append(qmefp_exrp)
                            break
            elif 'step' not in mol_id and mol_id.split('_')[1] not in folders:
                with open(f'{path}/G_LibEFP/ElecPot/Dimers/{mol_id}.in.out') as fl:
                    lns = fl.readlines()
                    for j1 in lns:
                        if 'DISPERSION ENERGY' in j1:
                            libefp_disp = float(j1.split()[-1]) 
                            lib_disp_values[f'{mol_id}'] = libefp_disp
                            libefp_dispersion.append(libefp_disp)
                        if 'EXCHANGE REPULSION ENERGY' in j1:
                            libefp_exrp = float(j1.split()[-1])
                            lib_ex_values[f'{mol_id}'] = libefp_exrp
                            libefp_exchangerep.append(libefp_exrp)
                            break
                with open(f'{path}/H_QMEFP/Dimers/{mol_id}_qmefp.in.out') as fl2:
                    lns2 = fl2.readlines()
                    for j2 in lns2:
                        if 'DISPERSION ENERGY' in j2:
                            qmefp_disp = float(j2.split()[-1])
                            qm_disp_values[f'{mol_id}'] = qmefp_disp
                            qmefp_dispersion.append(qmefp_disp)
                        if 'EXCHANGE-REPULSION ENERGY' in j2:
                            qmefp_exrp = float(j2.split()[-1])
                            qm_ex_values[f'{mol_id}'] = qmefp_exrp
                            qmefp_exchangerep.append(qmefp_exrp)
                            break         
            elif 'step' not in mol_id and mol_id.split('_')[1] in folders:
                with open(f'{path}/G_LibEFP/ElecPot/mixed_min2/{mol_id}.in.out') as fl:
                    lns = fl.readlines()
                    for j1 in lns:
                        if 'DISPERSION ENERGY' in j1:
                            libefp_disp = float(j1.split()[-1]) 
                            lib_disp_values[f'{mol_id}'] = libefp_disp
                            libefp_dispersion.append(libefp_disp)
                        if 'EXCHANGE REPULSION ENERGY' in j1:
                            libefp_exrp = float(j1.split()[-1])
                            lib_ex_values[f'{mol_id}'] = libefp_exrp
                            libefp_exchangerep.append(libefp_exrp)
                            break
                with open(f'{path}/H_QMEFP/mixed_min2/{mol_id}_qmefp.in.out') as fl2:
                    lns2 = fl2.readlines()
                    for j2 in lns2:
                        if 'DISPERSION ENERGY' in j2:
                            qmefp_disp = float(j2.split()[-1])
                            qm_disp_values[f'{mol_id}'] = qmefp_disp
                            qmefp_dispersion.append(qmefp_disp)
                        if 'EXCHANGE-REPULSION ENERGY' in j2:
                            qmefp_exrp = float(j2.split()[-1])
                            qm_ex_values[f'{mol_id}'] = qmefp_exrp
                            qmefp_exchangerep.append(qmefp_exrp)
                            break         
            else:    
                with open(f'{path}/G_LibEFP/ElecPot/{folder}/{mol_id}.out') as fl:
                    lns = fl.readlines()
                    for j1 in lns:
                        if 'DISPERSION ENERGY' in j1:
                            libefp_disp = float(j1.split()[-1]) 
                            lib_disp_values[f'{mol_id}'] = libefp_disp
                            libefp_dispersion.append(libefp_disp)
                        if 'EXCHANGE REPULSION ENERGY' in j1:
                            libefp_exrp = float(j1.split()[-1])
                            lib_ex_values[f'{mol_id}'] = libefp_exrp
                            libefp_exchangerep.append(libefp_exrp)
                            break
                with open(f'{path}/H_QMEFP/{folder}/{mol_id}_qmefp.out') as fl2:
                    lns2 = fl2.readlines()
                    for j2 in lns2:
                        if 'DISPERSION ENERGY' in j2:
                            qmefp_disp = float(j2.split()[-1])
                            qm_disp_values[f'{mol_id}'] = qmefp_disp
                            qmefp_dispersion.append(qmefp_disp)
                        if 'EXCHANGE-REPULSION ENERGY' in j2:
                            qmefp_exrp = float(j2.split()[-1])
                            qm_ex_values[f'{mol_id}'] = qmefp_exrp
                            qmefp_exchangerep.append(qmefp_exrp)
                            break
            total_inter = (libefp_disp + libefp_exrp) - (qmefp_disp + qmefp_exrp)
            total_values[f'{mol_id}'] = total_inter
            disp_exch_interaction.append(total_inter)

df['libefp_disp'] = libefp_dispersion
df['libefp_exrp'] = libefp_exchangerep
df['qmefp_disp'] = qmefp_dispersion
df['qmefp_exrp'] = qmefp_exchangerep
df['fixed_inter'] = disp_exch_interaction

df.to_csv('claudia_dataset_dispex_inter.csv', index = False, header = True)
