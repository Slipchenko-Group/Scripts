from __future__ import print_function
import sys

count = -1
global_count = 0
with open('./mc_run.log','r') as log:
	while True:
		
		line = log.readline()
		if not line:
			break
		if 'COORDINATES OF FRAGMENT MULTIPOLE CENTERS' in line:
			line = log.readline()
			line = log.readline()
			line = log.readline()
			line = log.readline()
			count = count + 1 
			if count %10 != 0:
				continue
			ref_fname = './ref/ref_ss'+str(global_count)+'.inp'
			sat_fname = './sat/sat_ss'+str(global_count)+'.inp'
			global_count = global_count + 1

			ref = open(ref_fname,'w')
			sat = open(sat_fname,'w')
			

			print('run_type energy',file=ref)
			print('terms elec pol disp xr',file=ref)
			print('elec_damp screen',file=ref)
			print('coord points',file=ref)
			print('pol_damp tt',file=ref)
			print('disp_damp tt',file=ref)

			print('run_type energy',file=sat)
			print('terms elec pol disp xr',file=sat)
			print('elec_damp screen',file=sat)
			print('coord points',file=sat)
			print('pol_damp tt',file=sat)
			print('disp_damp tt',file=sat)
			print('',file=ref)
			print('',file=sat)



			print('fragment F1REF',file=ref)
			print('fragment F1SAT',file=sat)
			print(line[5:],end='',file=ref)
			print(line[5:],end='',file=sat)
			line = log.readline()
			print(line[5:],end='',file=ref)
			print(line[5:],end='',file=sat)
			line = log.readline()
			print(line[5:],end='',file=ref)
			print(line[5:],end='',file=sat)


			while not line.startswith(' FRAGNAME'):
				line = log.readline()
			print('fragment BA',file=ref)
			print('fragment BA',file=sat)
			line = log.readline()
			print(line[5:],end='',file=ref)
			print(line[5:],end='',file=sat)
			line = log.readline()
			print(line[5:],end='',file=ref)
			print(line[5:],end='',file=sat)
			line = log.readline()
			print(line[5:],end='',file=ref)
			print(line[5:],end='',file=sat)

			ref.close()
			sat.close()