import os
import subprocess
# this script goes through each of the snapshots from andres' ligands and generates a qchem single point gas phase job file, then submits it with qc53
snapshots = [fl for fl in os.listdir(os.getcwd()) if fl.startswith('snp')]
atom_types = {8:'O',7:'N',6:'C',1:'H'}
path = os.getcwd()
rem = '''$rem
   JOBTYPE              sp
   METHOD               wb97x
   BASIS                6-31g*
   SCF_CONVERGENCE      5
   MAX_SCF_CYCLES       200
 $end''' 
for s in snapshots:
    os.chdir(f'{path}/{s}')
    with open('lig.efp','r') as fl:
        efp_lines = fl.readlines()
        for idx, ln in enumerate(efp_lines):
            if 'COORDINATES (BOHR)' in ln:
                start = idx
            if 'STOP' in ln:
                end = idx 
                break
        coords = efp_lines[start+1:end]
    f = open(f'{s}_gas.in','w+')
    f.write('$molecule\n')
    f.write('   0  1\n')
    for i in coords:
        atom = atom_types[int(float(i.split()[-1]))]
        x = float(i.split()[1])
        y = float(i.split()[2])
        z = float(i.split()[3])
        f.write(f'   {atom}   {x}   {y}   {z}\n')
    f.write('$end\n')
    f.write(f'{rem}')
    f.close()
    bashCommand = f'qc53 -q standby -w 4:00:00 -ccp 24 {s}_gas.in' 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    os.chdir('../')
