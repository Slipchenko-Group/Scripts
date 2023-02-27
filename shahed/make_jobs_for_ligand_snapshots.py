import os
import shutil
import subprocess

# this script is made to generate a bunch of qm/mm jobs for adres' ligands
# each snapshot needs its own directory because each one prints out its own grad.xyz among other files
# need to prepare all the .mdp and topols before hand and keep them in a separate directory

ligands = [fl for fl in os.listdir(os.getcwd()) if fl.startswith('snp')]
path = os.getcwd()


for l in ligands:
    os.chdir(f'{path}/{l}')
    snaps = [fl for fl in os.listdir(os.getcwd()) if fl.endswith('.g96')]
    for s in snaps:
        name = s.replace('.g96', '')
        ligand = l.replace('snp_','')
        try:
            os.mkdir(f'{path}/{l}/{name}')
        except:
            print(f'directory {path}/{l}/{name} already exists, moving on')
        shutil.copyfile(f'{s}',f'{name}/{s}')
        job_files = [fl for fl in os.listdir(f'{path}/files/{ligand}')]
        for j in job_files:
            shutil.copyfile(f'{path}/files/{ligand}/{j}',f'{name}/{j}')
        try:
            os.chmod(f'{path}/{l}/{name}/submit.sh', 0o775)
            os.chmod(f'{path}/{l}/{name}/make_tpr.sh', 0o775)
        except:
            print('job(s) does not exist')
        os.chdir(f'{path}/{l}/{name}/')
        subprocess.run([f'{path}/{l}/{name}/make_tpr.sh'], shell=True)
        subprocess.run([f'{path}/{l}/{name}/submit.sh'], shell=True)
        os.chdir(f'{path}/{l}/')
