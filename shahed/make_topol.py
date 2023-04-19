import os
import shutil

path = os.getcwd()
os.chdir(f'{path}/md')
mds = [fl for fl in os.listdir(os.getcwd())]

for folder in mds:
    fragment = folder.split('_')[0]
    solvent = folder.split('_')[-1]
    os.chdir(f'{path}/md/{folder}')
    shutil.copy(f'{path}/top/topol.top',f'{path}/md/{folder}/topol.top')
    itps = [fl for fl in os.listdir(os.getcwd()) if fl.endswith('itp')]
    prm = open('nonbonded.prm', 'w+')
    if 'frag' in itps[0]:
        with open(itps[0]) as frag_fl:
            frag_lns = frag_fl.readlines()
        if solvent != 'water':
            with open(itps[-1]) as solv_fl:
                solv_lns = solv_fl.readlines()
    else:
        with open(itps[-1]) as frag_fl:
            frag_lns = frag_fl.readlines()
        if solvent != 'water':
            with open(itps[0]) as solv_fl:
                solv_lns = solv_fl.readlines()
    if solvent == 'water':
        solv_lns = []
    atomtypes = []
    for idx, ln in enumerate(frag_lns):
        if '[ atomtypes ]\n' in ln:
            for ln2 in frag_lns[idx:]:
                if len(ln2.split()) == 10:
                    atomtypes.append(ln2)
                if len(ln2.split()) == 0:
                    break
            break
    if len(solv_lns) != 0:
        for idxx, lnn in enumerate(solv_lns):
            if '[ atomtypes ]\n' in lnn:
                for lnn2 in solv_lns[idxx:]:
                    if len(lnn2.split()) == 10 and lnn2 not in atomtypes:
                        atomtypes.append(lnn2)
                    if len(lnn2.split()) == 0:
                        break
                break
    prm.write('[ atomtypes ]\n')
    prm.write(';   name  at_num     mass   charge  type        sigma      epsilon\n')
    for a in atomtypes:
        prm.write(a)
    prm.write('\n')
    prm.close()
    frag_new_itp = open('frag_temp.itp', 'w+')
    if solvent != 'water':
        solv_new_itp = open('solv_temp.itp', 'w+')
    for idx, ln in enumerate(frag_lns):
        if '[ moleculetype ]' in ln:
            for ln2 in frag_lns[idx:]:
                frag_new_itp.write(ln2)
            frag_new_itp.close()
            break
    if solvent != 'water':
        for idxx, lnn in enumerate(solv_lns):
            if '[ moleculetype ]' in lnn:
                for lnn2 in solv_lns[idxx:]:
                    solv_new_itp.write(lnn2)
                solv_new_itp.close()
                break
    os.remove(f'{path}/md/{folder}/{fragment}_qforce_resp.itp')
    if solvent != 'water':
        os.remove(f'{path}/md/{folder}/{solvent}_qforce_resp.itp')
    os.rename(f'frag_temp.itp',f'{fragment}_qforce_resp.itp')
    if solvent != 'water':
        os.rename(f'solv_temp.itp',f'{solvent}_qforce_resp.itp')
    with open('topol.top') as fl:
        top_lns = fl.readlines()
        out = open(f'{fragment}_{solvent}.top', 'w+')
        for ln in top_lns:
            if ';       This is the template file for my topologies ' in ln:
                out.write(f';       This is the topology file for {fragment} in {solvent}\n')
            elif ';include_itp' in ln:
                out.write(f'#include "{fragment}_qforce_resp.itp"\n')
                if solvent != 'water':
                    out.write(f'#include "{solvent}_qforce_resp.itp"\n')
                if solvent == 'water':
                    out.write(f'include "amber03.ff/tip3p.itp"')
            elif ';insert_system_name' in ln:
                out.write(f'{fragment} in {solvent}\n')
            elif ';insert_fragment' in ln:
                out.write(f'{fragment}  1\n')
            elif ';insert_solvent' in ln and solvent != 'water':
                out.write(f'{solvent}   600\n')
            elif ';insert_solvent' in ln and solvent == 'water':
                out.write('SOL  600')
            else:
                out.write(ln)
        out.close()
        os.remove(f'{path}/md/{folder}/topol.top')
        os.rename(f'{fragment}_{solvent}.top','topol.top')


os.chdir(f'{path}')
