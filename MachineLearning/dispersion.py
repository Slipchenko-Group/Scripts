from __future__ import print_function
# from keras import callbacks
# from keras.models import Model
# from keras.models import Sequential
# from keras.optimizers import Adam
# from keras.layers import Dense, Activation, Input, Dropout, BatchNormalization, GaussianDropout
# from keras import optimizers
import itertools
import numpy as np
import os.path
from sklearn import preprocessing
from detect_bonds import Detect_bonds
from sklearn.externals import joblib
import lmo_arrange as la
import pickle
import time
from sklearn.neighbors import KNeighborsRegressor
from molml.features import CoulombMatrix


def calculate_bond_angles(a,b,c):
	ba = a-b
	bc = c-b
	cosine_angle = np.dot(ba,bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
	angle = np.arccos(cosine_angle)
	#return angle
	return np.degrees(angle)

def pick_atoms(a,bonds,cartesian):
	for i in range(len(bonds)):
		if bonds[a][i]==True or bonds[i][a]==True:
			for j in range(len(bonds)):
				if j == i or j == a:
					continue
				if bonds[i][j]==True or bonds[j][i]==True:
					if calculate_bond_angles(cartesian[a],cartesian[i],cartesian[j])!=180:
						return i,j

def pick_atom_for_bond(a,b,bonds,cartesian):
	for i in range(len(bonds)):
		if i==a or i==b:
			continue

		if bonds[b][i]==True or bonds[i][b]==True:
			if calculate_bond_angles(cartesian[a],cartesian[b],cartesian[i])!=180:
				return i

	for i in range(len(bonds)):
		if i==a or i==b:
			continue

		if bonds[a][i]==True or bonds[i][a]==True:
			if calculate_bond_angles(cartesian[b],cartesian[a],cartesian[i])!=180:
				return i


with open('./ref/ref','rb') as ref_data:
	metadata = pickle.load(ref_data)


atom_num = metadata['atom_num']
bond_num = metadata['bond_num']
lmo_count = metadata['lmo_count']
bonds = metadata['bonds']
ref_geo = metadata['ref_geo']
ref_lmo = metadata['ref_lmo']
ref_lmo_log = metadata['ref_lmo_log']
#inp_no = atom_num*atom_num
#inp_no = metadata['input_no']
inp_no = atom_num * 3
charge_sum = metadata['charge_sum']
dynamic_pol_suffix = metadata['dynamic_pol_suffix']
total_mass = metadata['total_mass']
masses = {"C":12.0107, "O":15.199, "N":14.0067, "H":1.00794}
	

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
						if digits[1]==bonds[i][0] and digits[0]==bonds[i][1]:
							bond_index = i
							break
				else:
					bond_index = -1
					for i in range(bond_num):
						if digits[0]==bonds[i][0] and digits[1]==bonds[i][1]:
							bond_index = i
							break
				curr_index += 1
			elif log_line.startswith('  LMO ALPHA POLARIZABILITY TENSOR FOR CORE OR LONE PAIR ON ATOM'):
				digits = [int(s) for s in log_line.split() if s.isdigit()]
				curr_index+=1
			if curr_index == lmo_count:
				break
		break


lmo_pol = np.zeros((7757,12*lmo_count*9))

current_index = 0
num_inp = 0
total_count = 0
count = 0


for filename in os.listdir('samples/'):
	if not filename.endswith('.efp'):
		continue
	with open('./samples/' + filename,'r') as f:

		curr_lmo_geo = []
		curr_index = 0
		temp_pol = np.zeros((12,lmo_count,9))
		while True:
			line = f.readline()
			if not line:
				break
			

			if line.startswith(' DYNAMIC POLARIZABLE POINTS'):
				lmo_index = 0
				while True:
					line = f.readline()
					if line.startswith(' STOP'):
						break
					if line.startswith('CT'):
						curr_index += 1
						tokens = line.split()
						curr_lmo_geo.append([float(tokens[2]),float(tokens[3]),float(tokens[4])])
						line = f.readline()
						tokens = line.split()
						temp_pol[lmo_index][curr_index-1][0] = float(tokens[0])
						temp_pol[lmo_index][curr_index-1][1] = float(tokens[1])
						temp_pol[lmo_index][curr_index-1][2] = float(tokens[2])
						temp_pol[lmo_index][curr_index-1][3] = float(tokens[3])
						line = f.readline()
						tokens = line.split()
						temp_pol[lmo_index][curr_index-1][4] = float(tokens[0])
						temp_pol[lmo_index][curr_index-1][5] = float(tokens[1])
						temp_pol[lmo_index][curr_index-1][6] = float(tokens[2])
						temp_pol[lmo_index][curr_index-1][7] = float(tokens[3])
						line = f.readline()
						tokens = line.split()
						temp_pol[lmo_index][curr_index-1][8] = float(tokens[0])
						if curr_index == lmo_count:
							curr_index = 0
							lmo_index += 1

				break

		db = Detect_bonds('./samples/' + filename)
		cartesian = db.get_cartesian()
		bond_list = []
		for i in range(atom_num):
			for j in range(atom_num):
				if i < j:
					if bonds[i][j] == True:
						bond_list.append([i,j])
	
		lmo = []
		for i in range(atom_num):
			lmo.append([])
		for i in range(bond_num):
			lmo.append([])
		curr_index = 0

		with open('./samples_log/'+filename[0:-4]+'.log','r') as log:
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
							lmo[atom_num+bond_index].append(curr_index)
						else:
							bond_index = -1
							for i in range(bond_num):
								if digits[0]==bond_list[i][0] and digits[1]==bond_list[i][1]:
									bond_index = i
									break
							lmo[atom_num+bond_index].append(curr_index)
						curr_index += 1
					elif log_line.startswith('  LMO ALPHA POLARIZABILITY TENSOR FOR CORE OR LONE PAIR ON ATOM'):
						digits = [int(s) for s in log_line.split() if s.isdigit()]
						lmo[digits[0]-1].append(curr_index)
						curr_index+=1
					if curr_index == lmo_count:
						break
				break
		final_lmo_geo = []
		final_lmo_pol = np.zeros((12,lmo_count,9))
		input_indicator = []
		current_index = 0

		for i in range(len(lmo)):
			if len(lmo[i]) == 1:
				final_lmo_geo.append(curr_lmo_geo[lmo[i][0]])
				for j in range(12):
					final_lmo_pol[j][current_index] = temp_pol[j][lmo[i][0]]
				current_index += 1

				#final_lmo_pol.append(temp_pol[lmo[i][0]].flatten().tolist())
				

			elif len(lmo[i]) > 1:

				if i < atom_num:
					a,b = pick_atoms(i,bonds,cartesian)
					to_xyz = np.zeros((3+len(lmo[i]),3))
					to_xyz[0]=cartesian[i]
					to_xyz[1]=cartesian[a]
					to_xyz[2]=cartesian[b]
					for k in range(3,len(to_xyz)):
						to_xyz[k]=np.asarray(curr_lmo_geo[lmo[i][k-3]])
					ref_xyz = np.zeros((3+len(lmo[i]),3))
					ref_xyz[0]=np.asarray(ref_geo[i])
					ref_xyz[1]=np.asarray(ref_geo[a])
					ref_xyz[2]=np.asarray(ref_geo[b])
					for k in range(3,len(ref_xyz)):
						ref_xyz[k]=np.asarray(ref_lmo[ref_lmo_log[i][k-3]])
					order = la.arrange(np.copy(ref_xyz),np.copy(to_xyz))
					for k in range(len(order)):
						final_lmo_geo.append(curr_lmo_geo[lmo[i][order[k][1]]])

						for m in range(12):
							final_lmo_pol[m][current_index] = temp_pol[m][lmo[i][order[k][1]]]
						current_index += 1
						#final_lmo_pol.append(temp_pol[lmo[i][order[k][1]]].flatten().tolist())

						input_indicator.append([1,i,-1])
				else:
					a = pick_atom_for_bond(bond_list[i-atom_num][0],bond_list[i-atom_num][1],bonds,cartesian)
					to_xyz = np.zeros((3+len(lmo[i]),3))
					to_xyz[0]=cartesian[bond_list[i-atom_num][0]]
					to_xyz[1]=cartesian[bond_list[i-atom_num][1]]
					to_xyz[2]=cartesian[a]
					for k in range(3,len(to_xyz)):
						to_xyz[k]=np.asarray(curr_lmo_geo[lmo[i][k-3]])
					ref_xyz = np.zeros((3+len(lmo[i]),3))
					ref_xyz[0]=np.asarray(ref_geo[bond_list[i-atom_num][0]])
					ref_xyz[1]=np.asarray(ref_geo[bond_list[i-atom_num][1]])
					ref_xyz[2]=np.asarray(ref_geo[a])
					for k in range(3,len(ref_xyz)):
						ref_xyz[k]=np.asarray(ref_lmo[ref_lmo_log[i][k-3]])
					order = la.arrange(ref_xyz,to_xyz)
					for k in range(len(order)):
						final_lmo_geo.append(curr_lmo_geo[lmo[i][order[k][1]]])
						for m in range(12):
							final_lmo_pol[m][current_index] = temp_pol[m][lmo[i][order[k][1]]]
						current_index += 1


						#final_lmo_pol.append(temp_pol[lmo[i][order[k][1]]].flatten().tolist())
		lmo_pol[total_count] = final_lmo_pol.flatten()
		total_count += 1



test_lmo_pol = np.zeros((99,12*lmo_count*9))
test_lmo_geo = np.zeros((99,lmo_count*3))
current_index = 0
total_count = 0
count = 0

for filename in os.listdir('test/'):
	if not filename.endswith('.efp'):
		continue
	with open('./test/' + filename,'r') as f:
		print(filename)
		curr_lmo_geo = []
		curr_index = 0
		temp_pol = np.zeros((12,lmo_count,9))
		while True:
			line = f.readline()
			if not line:
				break
			if line.startswith(' POLARIZABLE POINTS'):
				while True:
					line = f.readline()
					if line.startswith(' STOP'):
						break
					if line.startswith('CT'):
						curr_index += 1
						tokens = line.split()
						curr_lmo_geo.append([float(tokens[1]),float(tokens[2]),float(tokens[3])])

			if line.startswith(' DYNAMIC POLARIZABLE POINTS'):
				curr_index = 0
				lmo_index = 0
				while True:
					line = f.readline()
					if line.startswith(' STOP'):
						break
					if line.startswith('CT'):
						curr_index += 1
						tokens = line.split()
						line = f.readline()
						tokens = line.split()
						temp_pol[lmo_index][curr_index-1][0] = float(tokens[0])
						temp_pol[lmo_index][curr_index-1][1] = float(tokens[1])
						temp_pol[lmo_index][curr_index-1][2] = float(tokens[2])
						temp_pol[lmo_index][curr_index-1][3] = float(tokens[3])
						line = f.readline()
						tokens = line.split()
						temp_pol[lmo_index][curr_index-1][4] = float(tokens[0])
						temp_pol[lmo_index][curr_index-1][5] = float(tokens[1])
						temp_pol[lmo_index][curr_index-1][6] = float(tokens[2])
						temp_pol[lmo_index][curr_index-1][7] = float(tokens[3])
						line = f.readline()
						tokens = line.split()
						temp_pol[lmo_index][curr_index-1][8] = float(tokens[0])
						if curr_index == lmo_count:
							curr_index = 0
							lmo_index += 1

				break

		db = Detect_bonds('./test/' + filename)
		cartesian = db.get_cartesian()
		bond_list = []
		for i in range(atom_num):
			for j in range(atom_num):
				if i < j:
					if bonds[i][j] == True:
						bond_list.append([i,j])
	
		lmo = []
		for i in range(atom_num):
			lmo.append([])
		for i in range(bond_num):
			lmo.append([])
		curr_index = 0

		with open('./samples_log/'+filename[0:-4]+'.log','r') as log:
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
							lmo[atom_num+bond_index].append(curr_index)
						else:
							bond_index = -1
							for i in range(bond_num):
								if digits[0]==bond_list[i][0] and digits[1]==bond_list[i][1]:
									bond_index = i
									break
							lmo[atom_num+bond_index].append(curr_index)
						curr_index += 1
					elif log_line.startswith('  LMO ALPHA POLARIZABILITY TENSOR FOR CORE OR LONE PAIR ON ATOM'):
						digits = [int(s) for s in log_line.split() if s.isdigit()]
						lmo[digits[0]-1].append(curr_index)
						curr_index+=1
					if curr_index == lmo_count:
						break
				break
		final_lmo_geo = []
		final_lmo_pol = np.zeros((12,lmo_count,9))
		input_indicator = []
		current_index = 0

		for i in range(len(lmo)):
			if len(lmo[i]) == 1:
				final_lmo_geo.append(curr_lmo_geo[lmo[i][0]])
				for j in range(12):
					final_lmo_pol[j][current_index] = temp_pol[j][lmo[i][0]]
				current_index += 1

				#final_lmo_pol.append(temp_pol[lmo[i][0]].flatten().tolist())
				

			elif len(lmo[i]) > 1:

				if i < atom_num:
					a,b = pick_atoms(i,bonds,cartesian)
					to_xyz = np.zeros((3+len(lmo[i]),3))
					to_xyz[0]=cartesian[i]
					to_xyz[1]=cartesian[a]
					to_xyz[2]=cartesian[b]
					for k in range(3,len(to_xyz)):
						to_xyz[k]=np.asarray(curr_lmo_geo[lmo[i][k-3]])
					ref_xyz = np.zeros((3+len(lmo[i]),3))
					ref_xyz[0]=np.asarray(ref_geo[i])
					ref_xyz[1]=np.asarray(ref_geo[a])
					ref_xyz[2]=np.asarray(ref_geo[b])
					for k in range(3,len(ref_xyz)):
						ref_xyz[k]=np.asarray(ref_lmo[ref_lmo_log[i][k-3]])
					order = la.arrange(np.copy(ref_xyz),np.copy(to_xyz))
					for k in range(len(order)):
						final_lmo_geo.append(curr_lmo_geo[lmo[i][order[k][1]]])

						for m in range(12):
							final_lmo_pol[m][current_index] = temp_pol[m][lmo[i][order[k][1]]]
						current_index += 1
						#final_lmo_pol.append(temp_pol[lmo[i][order[k][1]]].flatten().tolist())

						input_indicator.append([1,i,-1])
				else:
					a = pick_atom_for_bond(bond_list[i-atom_num][0],bond_list[i-atom_num][1],bonds,cartesian)
					to_xyz = np.zeros((3+len(lmo[i]),3))
					to_xyz[0]=cartesian[bond_list[i-atom_num][0]]
					to_xyz[1]=cartesian[bond_list[i-atom_num][1]]
					to_xyz[2]=cartesian[a]
					for k in range(3,len(to_xyz)):
						to_xyz[k]=np.asarray(curr_lmo_geo[lmo[i][k-3]])
					ref_xyz = np.zeros((3+len(lmo[i]),3))
					ref_xyz[0]=np.asarray(ref_geo[bond_list[i-atom_num][0]])
					ref_xyz[1]=np.asarray(ref_geo[bond_list[i-atom_num][1]])
					ref_xyz[2]=np.asarray(ref_geo[a])
					for k in range(3,len(ref_xyz)):
						ref_xyz[k]=np.asarray(ref_lmo[ref_lmo_log[i][k-3]])
					order = la.arrange(ref_xyz,to_xyz)
					for k in range(len(order)):
						final_lmo_geo.append(curr_lmo_geo[lmo[i][order[k][1]]])
						for m in range(12):
							final_lmo_pol[m][current_index] = temp_pol[m][lmo[i][order[k][1]]]
						current_index += 1
		for i in range(lmo_count):
			test_lmo_geo[total_count][i*3] = final_lmo_geo[i][0]
			test_lmo_geo[total_count][i*3+1] = final_lmo_geo[i][1]
			test_lmo_geo[total_count][i*3+2] = final_lmo_geo[i][2]
			


						#final_lmo_pol.append(temp_pol[lmo[i][order[k][1]]].flatten().tolist())
		test_lmo_pol[total_count] = final_lmo_pol.flatten()
		total_count += 1


input_list = np.zeros((7757,inp_no))
total_count = 0
count=0
for filename in os.listdir('./samples/'):
	if not filename.endswith('.efp'):
		continue;
	with open('./samples/' + filename,'r') as f:

		# feat = CoulombMatrix(input_type='filename')

		# feat.fit(['./samples/'+filename+'.xyz'])
		# trans = feat.transform(['./samples/'+filename+'.xyz'])
		# for i in range(inp_no):
		# 	input_list[total_count][i] = trans[0][i]


		# db = Detect_bonds('./samples/' + filename)
		# bond_len = db.all_lengths(bonds)
		# bond_ang = db.all_bond_angles(bonds)
		# dih_ang = db.all_dihedral_angles(bonds)


		# loc_count = 0
		# for i in range(len(bond_len)):
		# 	input_list[total_count][loc_count] = bond_len[i]
		# 	loc_count = loc_count+1
		# for i in range(len(bond_ang)):  
		# 	input_list[total_count][loc_count] = bond_ang[i]
		# 	loc_count = loc_count+1
		# for i in range(len(dih_ang)):
		# 	input_list[total_count][loc_count] = dih_ang[i]
		# 	loc_count = loc_count+1
		xyz = open('./samples/'+filename+'.xyz', 'r')
		xyz.readline()
		xyz.readline()
		count = 0
		atoms = []
		for i in range(atom_num):
			line = xyz.readline()
			tokens = line.split()
			atoms.append(tokens[0].upper())
			input_list[total_count][count] = float(tokens[1])
			count+=1
			input_list[total_count][count] = float(tokens[2])
			count+=1
			input_list[total_count][count] = float(tokens[3])
			count+=1


		x_sum = 0
		y_sum = 0
		z_sum = 0


		for i in range(atom_num):
			x_sum += input_list[total_count][3*i] * masses[atoms[i]]
		x_shift = x_sum/total_mass
		for i in range(atom_num):
			input_list[total_count][3*1] -= x_shift


		for i in range(atom_num):
			y_sum += input_list[total_count][3*i+1] * masses[atoms[i]]
		y_shift = y_sum/total_mass
		for i in range(atom_num):
			input_list[total_count][3*1+1] -= y_shift


		for i in range(atom_num):
			z_sum += input_list[total_count][3*i+2] * masses[atoms[i]]
		z_shift = z_sum/total_mass
		for i in range(atom_num):
			input_list[total_count][3*1+2] -= z_shift


		total_count = total_count+1
	
	

test_input_list = np.zeros((99,inp_no))
total_count = 0
count=0
for filename in os.listdir('./test/'):
	if not filename.endswith('.efp'):
		continue;
	with open('./test/' + filename,'r') as f:

		# feat = CoulombMatrix(input_type='filename')
		# feat.fit(['./test/'+filename+'.xyz'])
		# trans = feat.transform(['./test/'+filename+'.xyz'])
		# for i in range(inp_no):
		# 	test_input_list[total_count][i] = trans[0][i]

		# db = Detect_bonds('./test/' + filename)
		# bond_len = db.all_lengths(bonds)
		# bond_ang = db.all_bond_angles(bonds)
		# dih_ang = db.all_dihedral_angles(bonds)

		
		# loc_count = 0
		# for i in range(len(bond_len)):
		# 	test_input_list[total_count][loc_count] = bond_len[i]
		# 	loc_count = loc_count+1
		# for i in range(len(bond_ang)):
		# 	test_input_list[total_count][loc_count] = bond_ang[i]
		# 	loc_count = loc_count+1
		# for i in range(len(dih_ang)):
		# 	test_input_list[total_count][loc_count] = dih_ang[i]
		# 	loc_count = loc_count+1

		xyz = open('./test/'+filename+'.xyz', 'r')
		xyz.readline()
		xyz.readline()
		count = 0
		for i in range(atom_num):
			line = xyz.readline()
			tokens = line.split()
			test_input_list[total_count][count] = float(tokens[1])
			count+=1
			test_input_list[total_count][count] = float(tokens[2])
			count+=1
			test_input_list[total_count][count] = float(tokens[3])
			count+=1



		x_sum = 0
		y_sum = 0
		z_sum = 0


		for i in range(atom_num):
			x_sum += test_input_list[total_count][3*i] * masses[atoms[i]]
		x_shift = x_sum/total_mass
		for i in range(atom_num):
			test_input_list[total_count][3*1] -= x_shift


		for i in range(atom_num):
			y_sum += test_input_list[total_count][3*i+1] * masses[atoms[i]]
		y_shift = y_sum/total_mass
		for i in range(atom_num):
			test_input_list[total_count][3*1+1] -= y_shift


		for i in range(atom_num):
			z_sum += test_input_list[total_count][3*i+2] * masses[atoms[i]]
		z_shift = z_sum/total_mass
		for i in range(atom_num):
			test_input_list[total_count][3*1+2] -= z_shift
		total_count = total_count+1
		



# x_minmaxscaler = joblib.load('Trained/inp_minmax.pkl')
# x_train_scaled = x_minmaxscaler.transform(input_list)


# y_standardscaler = preprocessing.StandardScaler().fit(lmo_pol)
# y_train_scaled = y_standardscaler.transform(lmo_pol)
# y_minmaxscaler = preprocessing.MinMaxScaler((-1,1)).fit(y_train_scaled)
# y_train_scaled = y_minmaxscaler.transform(y_train_scaled)

# joblib.dump(x_standardscaler,'./Trained/inp_std.pkl')
# joblib.dump(x_minmaxscaler,'./Trained/inp_minmax.pkl')

# joblib.dump(y_standardscaler,'./Trained/dispersion_std.pkl')
# joblib.dump(y_minmaxscaler,'./Trained/dispersion_minmax.pkl')


model = KNeighborsRegressor(n_neighbors = 3, weights = 'distance' , p = 1)
model.fit(input_list, lmo_pol)
joblib.dump(model, 'Trained/dynamic_pol.pkl')


# inputs = Input(shape=(inp_no,))
# hidden_layer = Dense(750,activation='relu')(inputs)
# hidden_layer = GaussianDropout(0.2)(hidden_layer)
# hidden_layer = Dense(750,activation='relu')(hidden_layer)
# hidden_layer = GaussianDropout(0.2)(hidden_layer)
# hidden_layer = Dense(750,activation='relu')(hidden_layer)
# hidden_layer = GaussianDropout(0.2)(hidden_layer)
# hidden_layer = Dense(750,activation='relu')(hidden_layer)
# hidden_layer = GaussianDropout(0.2)(hidden_layer)
# hidden_layer = Dense(750,activation='relu')(hidden_layer)
# hidden_layer = GaussianDropout(0.2)(hidden_layer)
# hidden_layer = Dense(750,activation='relu')(hidden_layer)




# output = Dense(12*lmo_count*9, activation='linear')(hidden_layer)
# adam = Adam(lr=0.0005)
# model = Model(input=inputs, output=output)
# model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mean_absolute_error'])
# model.fit(x_train_scaled, y_train_scaled,epochs=500, batch_size=64,verbose=1,shuffle=True,validation_split=0.1)

# model.save('./Trained/dispersion.h5')

def print_array_4(array,out):
	count = 0
	for i in range(int(len(array) / 4.0)):
		print(str(format(array[:][count],' .10f')) + '   ' + str(format(array[:][count+1],' .10f')) + '   ' 
			+ str(format(array[:][count+2],' .10f'))
			+ '   ' + str(format(array[:][count+3],' .10f')) , file=out)
		count += 4
	if len(array) % 4 == 1:
		print(str(format(array[:][count],' .10f')), file=out)
	elif len(array) % 4 == 2:
		print(str(format(array[:][count],' .10f')) + '   ' + str(format(array[:][count+1],' .10f')), file=out)
	elif len(array) % 4 == 3:
		print(str(format(array[:][count],' .10f')) + '   ' + str(format(array[:][count+1],' .10f')) + '   ' 
			+ str(format(array[:][count+2],' .10f')), file=out)
	

def print_array_3(array,out):
	count = 0
	for i in range(int(len(array) / 3.0)):
		print(str(format(array[:][count],' .10f')) + '   ' + str(format(array[:][count+1],' .10f')) + '   ' 
			+ str(format(array[:][count+2],' .10f'))
			 , file=out)
		count += 3
	if len(array) % 3 == 1:
		print(str(format(array[:][count],' .10f')), file=out)
	elif len(array) % 3 == 2:
		print(str(format(array[:][count],' .10f')) + '   ' + str(format(array[:][count+1],' .10f')), file=out)
	

# out = open('disp.out','w')

# for i in range(0,99):
# 	x = test_input_list[i]
# 	x = x_standardscaler.transform(x.reshape(1, -1))
# 	x = x_minmaxscaler.transform(x)
# 	y1 = model.predict(x.reshape(-1,inp_no))
	
# 	# y1 = y_minmaxscaler.inverse_transform(y1.reshape(1, -1))
# 	# y1 = y_standardscaler.inverse_transform(y1)
# 	print('Predicted', file=out)
# 	print_array_4(y1[0],out)
	
	
# 	print('Desired', file=out)
# 	print_array_4(test_lmo_pol[i],out)
	

# 	print('---------------------------------------------',file=out)

# out.close()











