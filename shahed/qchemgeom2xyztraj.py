import numpy as np
import pandas as pd
import os

# this script converts qchem geometry optimization .out to xyz format to be opened in vmd

# seems like you sometimes need to take the qchem .out file and copy paste it into another file
# fl.readlines() doesnt like original qchem .out file idk why

data = [fl for fl in os.listdir(os.getcwd()) if fl.endswith('.out') and 'slurm' not in fl]
path = os.getcwd()

for out_md in data:
    with open(out_md,'r') as fl:
        out_lns = fl.readlines()
        i = 1
        # here we just look for the number of atoms in the molecule/system
        for idx, ln in enumerate(out_lns):
            if 'Molecular Point Group ' in ln:
                num_atoms = int(out_lns[idx-2].split()[0])
                break
        # here we are grabbing each time steps geometries and putting them in an xyz file
        out_xyz = open(f'{path}/{out_md}_geom.xyz','w+')
        for idx, ln in enumerate(out_lns):
            if 'Optimization Cycle' in ln:
                step = int(out_lns[idx].split()[-1])
            if 'Standard Nuclear Orientation (Angstroms)' in ln:
                geom = out_lns[idx+3:idx+3+num_atoms]
                xyz = []
                # getting rid of the first column and putting it in xyz array
                for a in geom:
                   xyz.append('       '.join(a.split()[1:])) 
                # writing to a file
                out_xyz.write(f'{num_atoms}\n')
                out_xyz.write(f' OPTIMIZATION STEP #{i}\n')
                i+=1
                for a in xyz:
                    out_xyz.write(f' {a}\n')
        out_xyz.close()
