from __future__ import print_function
import os
import time
import os.path

no = 0
for name in os.listdir('./ref'):
	if name.endswith('.inp'):
		no += 1

while not os.path.exists('./ref'):
	print('ref...sleeping...')
	time.sleep(5)


while not os.path.exists('./sat'):
	print('sat...sleeping...')
	time.sleep(5)

while no != len([name for name in os.listdir('./ref/output/') if os.path.isfile('./ref/output/'+name)]):
	print('sleeping......')
	time.sleep(5)


while no != len([name for name in os.listdir('./sat/output/') if os.path.isfile('./sat/output/'+name)]):
	print('sleeping......')
	time.sleep(5)


overall_fname = './sat_results/overall.csv'


sat_elec_abs_fname = './sat_results/elec_abs.csv'
sat_elec_rel_fname = './sat_results/elec_rel.csv'

sat_pol_abs_fname = './sat_results/pol_abs.csv'
sat_pol_rel_fname = './sat_results/pol_rel.csv'

sat_rep_abs_fname = './sat_results/rep_abs.csv'
sat_rep_rel_fname = './sat_results/rep_rel.csv'

sat_disp_abs_fname = './sat_results/disp_abs.csv'
sat_disp_rel_fname = './sat_results/disp_rel.csv'

sat_total_abs_fname = './sat_results/total_abs.csv'
sat_total_rel_fname = './sat_results/total_rel.csv'




overall_f = open(overall_fname,'w')

sat_elec_abs_f = open(sat_elec_abs_fname,'w')
sat_elec_rel_f = open(sat_elec_rel_fname,'w')

sat_pol_abs_f = open(sat_pol_abs_fname,'w')
sat_pol_rel_f = open(sat_pol_rel_fname,'w')

sat_rep_abs_f = open(sat_rep_abs_fname,'w')
sat_rep_rel_f = open(sat_rep_rel_fname,'w')

sat_disp_abs_f = open(sat_disp_abs_fname,'w')
sat_disp_rel_f = open(sat_disp_rel_fname,'w')

sat_total_abs_f = open(sat_total_abs_fname,'w')
sat_total_rel_f = open(sat_total_rel_fname,'w')

for filename in os.listdir('./ref/output'):
	with open('./ref/output/'+filename,'r') as ref:
		with open('./sat/output/'+'sat_'+filename[4:],'r') as sat:
			ref_elec = 0
			ref_rep = 0
			ref_pol = 0
			ref_disp = 0
			ref_total = 0
			for line in ref:
				if 'ELECTROSTATIC ENERGY' in line:
					tokens = line.split()
					ref_elec = float(tokens[-1])
				elif 'EXCHANGE REPULSION ENERGY' in line:
					tokens = line.split()
					ref_rep = float(tokens[-1])
				elif 'POLARIZATION ENERGY' in line:
					tokens = line.split()
					ref_pol = float(tokens[-1])
				elif 'DISPERSION ENERGY' in line:
					tokens = line.split()
					ref_disp = float(tokens[-1])
				elif 'TOTAL ENERGY' in line:
					tokens = line.split()
					ref_total = float(tokens[-1])

			if ref_elec == 0 or ref_rep == 0 or ref_pol == 0 or ref_disp == 0 or ref_total == 0:
				continue
				
			sat_elec = 0
			sat_rep = 0
			sat_pol = 0
			sat_disp = 0
			sat_total = 0
			for line in sat:
				if 'ELECTROSTATIC ENERGY' in line:
					tokens = line.split()
					sat_elec = float(tokens[-1])
				elif 'EXCHANGE REPULSION ENERGY' in line:
					tokens = line.split()
					sat_rep = float(tokens[-1])
				elif 'POLARIZATION ENERGY' in line:
					tokens = line.split()
					sat_pol = float(tokens[-1])
				elif 'DISPERSION ENERGY' in line:
					tokens = line.split()
					sat_disp = float(tokens[-1])
				elif 'TOTAL ENERGY' in line:
					tokens = line.split()
					sat_total = float(tokens[-1])



			
			





			print(str(ref_elec)+','+str(sat_elec)+','+str(ref_pol)+','+str(sat_pol)+','+str(ref_total)+','+str(sat_total),file=overall_f)


			print(abs(float(ref_elec)-float(sat_elec))*627.5095,file=sat_elec_abs_f)
			print(abs(float(ref_elec)-float(sat_elec))*627.5095*100/(abs(float(ref_elec))*627.5095),file=sat_elec_rel_f)


			
			print(abs(float(ref_pol)-float(sat_pol))*627.5095,file=sat_pol_abs_f)
			print(abs(float(ref_pol)-float(sat_pol))*627.5095*100/(abs(float(ref_pol))*627.5095),file=sat_pol_rel_f)

			
			print(abs(float(ref_total)-float(sat_total))*627.5095,file=sat_total_abs_f)
			print(abs(float(ref_total)-float(sat_total))*627.5095*100/(abs(float(ref_total))*627.5095),file=sat_total_rel_f)


			if (ref_rep != 0):
				print(abs(float(ref_rep)-float(sat_rep))*627.5095,file=sat_rep_abs_f)
				print(abs(float(ref_rep)-float(sat_rep))*627.5095*100/(abs(float(ref_rep))*627.5095),file=sat_rep_rel_f)

			print(abs(float(ref_disp)-float(sat_disp))*627.5095,file=sat_disp_abs_f)
			print(abs(float(ref_disp)-float(sat_disp))*627.5095*100/(abs(float(ref_disp))*627.5095),file=sat_disp_rel_f)



			
sat_elec_abs_f.close()
sat_elec_rel_f.close()

sat_pol_abs_f.close()
sat_pol_rel_f.close()

sat_total_abs_f.close()
sat_total_rel_f.close()

sat_rep_abs_f.close()
sat_rep_rel_f.close()

sat_disp_rel_f.close()
sat_disp_abs_f.close()

overall_f.close()




