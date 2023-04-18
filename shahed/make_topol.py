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
    with open('topol.top') as fl:
        top_lns = fl.readlines()
        out = open(f'{fragment}_{solvent}.top', 'w+')
        for ln in top_lns:
            if ';       This is the template file for my topologies ' in ln:
                out.write(f';       This is the topology file for {fragment} in {solvent}\n')
            elif ';include_itp' in ln:
                out.write(f'#include "{fragment}_qforce_resp.itp\n')
                if solvent != 'water':
                    out.write(f'#include {solvent}_qforce_resp.itp\n')
            elif ';insert_system_name' in ln:
                out.write(f'{fragment} in {solvent}\n')
            elif ';insert_fragment' in ln:
                out.write(f'{fragment}  1\n')
            elif ';insert_solvent' in ln:
                out.write(f'{solvent}   600\n')
            else:
                out.write(ln)
        out.close()
        os.remove(f'{path}/md/{folder}/topol.top')
        os.rename(f'{fragment}_{solvent}.top','topol.top')


os.chdir(f'{path}')
