import os


sub_lns = """#!/bin/bash

#SBATCH -t 04:00:00

module load use.own
module load anaconda
conda activate nn_qmefp0

host=`hostname -s`

echo $CUDA_VISIBLE_DEVICES
cd $SLURM_SUBMIT_DIR
# Run on the first available GPU
"""

name = 'A1_dW_1A_SEP1_0BN0W'
py_fl = f'{name}.py'
with open(py_fl) as fl:
	py_lns = fl.readlines()

lr_factors = [0.5, 0.2, 0.1]
wdecays1 = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
wdecays2 = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]

fc = 1
for lr in lr_factors:
	for w1 in wdecays1:
		for w2 in wdecays2:
			out = open(f'{name}_{fc}.py', 'w+')
			for ln in py_lns:
				if '#insert_lr_f' in ln:
					out.write(f'lr_f = {lr}\n')
				elif '#insert_wdecay1' in ln:
					out.write(f'wdecay1 = {w1}\n')
				elif '#insert_wdecay2' in ln:
					out.write(f'wdecay2 = {w2}\n')
				elif '#insert_name' in ln:
					out.write(f'name = "{name}_{fc}"\n')
				else:
					out.write(ln)
			out.close()
			sub = open(f'{name}_{fc}.sub', 'w+')
			sub.write(sub_lns)
			sub.write(f'python {name}_{fc}.py > {name}_{fc}.out')
			sub.close()
			fc += 1
