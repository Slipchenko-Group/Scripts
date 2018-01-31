from __future__ import print_function
import pickle
import numpy as np
from detect_bonds import Detect_bonds

masses = {"C":12.0107, "O":15.999, "N":14.0067, "H":1.00794}
atom_num = 0
bond_num = 0
bonds = []
moments = []
lmo_count = 0
ref_geo = []
ref_lmo = []
ref_lmo_log = []
charge_sum = 0
charge_suffix = []
coord_suffix = []
dynamic_pol_suffix = []
projection_basis_set = []
multiplicity = ''
wave_per_lmo = 0
total_mass = 0





ref = open('./ref/ref.efp','r')
finish = False
while finish == False:
	line = ref.readline()
	if not line:
		break
	if line.startswith(' COORDINATES (BOHR)'):
		while True:
			
			line = ref.readline()
			if line.startswith(' STOP'):
				break
			tokens = line.split()
			coord_suffix.append(tokens[4])
			if line.startswith('A'):
				atom_num += 1
				ref_geo.append([float(tokens[1]),float(tokens[2]),float(tokens[3])])
			elif line.startswith('B'):
				bond_num += 1
				if len(tokens[0])==4:

					bonds.append([int(tokens[0][-2]),int(tokens[0][-1])])
				elif len(tokens[0])==5:
					bonds.append([int(tokens[0][-3:-1]),int(tokens[0][-1])])
				elif len(tokens[0])==6:
					bonds.append([int(tokens[0][-4:-2]),int(tokens[0][-2:])])
			moments.append(tokens[0])
	if line.startswith(' MONOPOLES'):
		while True:
			line = ref.readline()
			if line.startswith(' STOP'):
				break
			tokens = line.split()
			charge_sum += float(tokens[1])
			charge_suffix.append(tokens[2])

	if line.startswith(' POLARIZABLE POINTS'):
		while True:
			line = ref.readline()
			if line.startswith(' STOP'):
				
				break
			if line.startswith('CT'):
				lmo_count += 1
				tokens = line.split()
				ref_lmo.append([float(tokens[1]),float(tokens[2]),float(tokens[3])])
	if line.startswith(' DYNAMIC POLARIZABLE POINTS'):
		while True:
			line = ref.readline()
			if line.startswith(' STOP'):
				
				break
			if 'A.U.' in line:
				tokens = line.split()
				temp = ''
				for i in range(5,len(tokens)):
					temp = temp + ' ' + tokens[i]  
				dynamic_pol_suffix.append(temp)

	temp = []

	if line.startswith(' PROJECTION BASIS SET'):
		basis_set_count = 0
		while True:
			line = ref.readline()
			if line.startswith(' STOP'):
				break

			tokens = line.split()
			if not tokens:
				continue
			if tokens[0] in moments:
				if temp:
					projection_basis_set.append(temp)
				temp = []
				temp.append(tokens[4])
			else:
				temp.append(line)
		projection_basis_set.append(temp)

	if line.startswith( ' MULTIPLICITY'):
		tokens = line.split()
		multiplicity = int(tokens[-1])

	if line.startswith(' PROJECTION WAVEFUNCTION'):
		finish = True
		tokens = line.split()
		wave_per_lmo = int(tokens[-1])







complete_bonds = np.zeros((atom_num,atom_num))
for i in range(len(bonds)):
	complete_bonds[bonds[i][0]-1][bonds[i][1]-1] = True
	complete_bonds[bonds[i][1]-1][bonds[i][0]-1] = True


db = Detect_bonds('ref/ref.efp',atoms=atom_num)
bond_len = db.all_lengths(complete_bonds)
bond_ang = db.all_bond_angles(complete_bonds)
dih_ang = db.all_dihedral_angles(complete_bonds)
input_no = len(bond_len) + len(bond_ang) + len(dih_ang)
#input_no = len(bond_ang) + len(dih_ang)


bond_list = []
for i in range(atom_num):
	for j in range(atom_num):
		if i < j:
			if complete_bonds[i][j] == True:
				bond_list.append([i,j])


for i in range(atom_num):
	ref_lmo_log.append([])
for i in range(bond_num):
	ref_lmo_log.append([])
with open('ref/ref.log') as log:
	curr_index = 0
	while True:
		log_line = log.readline()
		if not log_line.startswith('           LOCALIZED ALPHA POLARIZABILITIES'):
			continue
		while True:
			log_line = log.readline()
			if log_line.startswith('  LMO ALPHA POLARIZABILITY TENSOR FOR BOND BETWEEN ATOM'):
				digits = [int(s) for s in log_line.split() if s.isdigit()]
				digits[0]-=1
				digits[1]-=1
				if digits[1]<digits[0]:
					bond_index = -1
					for i in range(bond_num):
						if digits[1]==bond_list[i][0] and digits[0]==bond_list[i][1]:
							bond_index = i
							break
					ref_lmo_log[atom_num+bond_index].append(curr_index)
				else:
					bond_index = -1
					for i in range(bond_num):
						if digits[0]==bond_list[i][0] and digits[1]==bond_list[i][1]:
							bond_index = i
							break
					ref_lmo_log[atom_num+bond_index].append(curr_index)
				curr_index += 1
			elif log_line.startswith('  LMO ALPHA POLARIZABILITY TENSOR FOR CORE OR LONE PAIR ON ATOM'):
				digits = [int(s) for s in log_line.split() if s.isdigit()]
				ref_lmo_log[digits[0]-1].append(curr_index)
				curr_index+=1
			if curr_index == lmo_count:
				break
		break


with open('./ref/ref.xyz','r') as xyz:
	xyz.readline()
	xyz.readline()
	for i in range(atom_num):
		line = xyz.readline()
		tokens = line.split()
		total_mass += masses[tokens[0].upper()]


			
print(total_mass)

metadata = {
	'ref_geo' : ref_geo,
	'ref_lmo' : ref_lmo,
	'ref_lmo_log' : ref_lmo_log,
	'atom_num' : atom_num,
	'bond_num' : bond_num,
	'lmo_count' : lmo_count,
	'mediums' : bonds,
	'bonds' : complete_bonds,
	'moments' : moments,
	'input_no' : input_no,
	'charge_sum': int(round(charge_sum)),
	'charge_suffix': charge_suffix,
	'coord_suffix': coord_suffix,
	'dynamic_pol_suffix': dynamic_pol_suffix,
	'projection_basis_set' : projection_basis_set,
	'wave_per_lmo': wave_per_lmo,
	'multiplicity': multiplicity,
	'total_mass': total_mass
}



with open('./ref/ref','w') as handle:
	pickle.dump(metadata,handle,protocol=pickle.HIGHEST_PROTOCOL)







