from __future__ import print_function
# from keras import callbacks
# from keras.models import Model
# from keras.optimizers import Adam
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Input, Dropout, BatchNormalization, GaussianDropout
# from keras import optimizers
import itertools
import numpy as np
import os.path
from sklearn import preprocessing
from detect_bonds import Detect_bonds
import lmo_arrange as la
import pickle
import time
import molml
from molml.features import CoulombMatrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib


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


samples_names = []
with open('./ref/ref','rb') as ref_data:
	metadata = pickle.load(ref_data)

masses = {"C":12.0107, "O":15.1156, "N":14.0067, "H":1.00794}
atom_num = metadata['atom_num']
bond_num = metadata['bond_num']
lmo_count = metadata['lmo_count']
bonds = metadata['bonds']
ref_geo = metadata['ref_geo']
ref_lmo = metadata['ref_lmo']
ref_lmo_log = metadata['ref_lmo_log']
#inp_no = metadata['input_no']
#inp_no = atom_num*atom_num
inp_no = atom_num * 3
charge_sum = metadata['charge_sum']
total_mass = metadata['total_mass']
	

multipole_mat = np.zeros((7757,atom_num+bond_num,19))
charge_map = np.zeros((7757,atom_num+bond_num))
current_index = 0
num_inp = 0
total_count = 0
count = 0


for filename in os.listdir('samples/'):
	if not filename.endswith('.efp'):
		continue
	with open('./samples/' + filename,'r') as f:
		samples_names.append(filename)
		curr_index = 0
		while True:
			line = f.readline()
			if not line:
				break
			if line.startswith(' MONOPOLES'):
				while True:
					line = f.readline()
					tokens = line.split()
					charge_map[total_count][count] = float(tokens[1])
					
					count += 1
					if count == atom_num+bond_num:
						
						count = 0
						break

			if line.startswith(' DIPOLES'):
				while True:
					line = f.readline()
					tokens = line.split()
					for i in range(3):
						multipole_mat[total_count][count][current_index] = float(tokens[i+1])
						current_index += 1
					current_index -= 3
					count += 1
					if count == atom_num+bond_num:
						current_index += 3
						count = 0
						break

			if line.startswith(' QUADRUPOLES'):
				while True:
					line = f.readline()
					tokens = line.split()
					if len(tokens)>2:
						for i in range(4):
							multipole_mat[total_count][count][current_index] = float(tokens[i+1])
							current_index += 1
					elif len(tokens)==2:
						for i in range(2):
							multipole_mat[total_count][count][current_index] = float(tokens[i])
							current_index += 1
						current_index -= 6
						count += 1
					if count == atom_num+bond_num:
						current_index += 6
						count = 0
						break


			if line.startswith(' OCTUPOLES'):
				while True:
					line=f.readline()
					tokens = line.split()
					if len(tokens)>5:
						for i in range(4):
							multipole_mat[total_count][count][current_index] = float(tokens[i+1])
							current_index += 1
					elif len(tokens)==5:
						for i in range(4):
							multipole_mat[total_count][count][current_index] = float(tokens[i])
							current_index += 1
					elif len(tokens)<5:
						for i in range(2):
							multipole_mat[total_count][count][current_index] = float(tokens[i])
							current_index += 1
						current_index -= 10
						count += 1
					if count == atom_num+bond_num:
						current_index = 0
						count = 0
						break

		total_count += 1

charge_map = charge_map / charge_sum
new_multipole_mat = np.zeros((7757,19*(atom_num+bond_num)))
total_count = 0
for i in range(7757):
	count = 0
	for j in range(atom_num+bond_num):
		for k in range(19):
			new_multipole_mat[total_count][count] = multipole_mat[total_count][j][k]
			count+=1
	total_count+=1


# test_charge_map = np.zeros((156,atom_num+bond_num))
# test_multipole_mat = np.zeros((156,atom_num+bond_num,19))
# current_index = 0
# total_count = 0
# count = 0



# for filename in os.listdir('crystal_test/'):
# 	if not filename.endswith('.efp'):
# 		continue
# 	with open('./crystal_test/' + filename,'r') as f:

# 		curr_lmo_geo = []
# 		curr_index = 0
# 		temp_pol = np.zeros((lmo_count,9))
# 		while True:
# 			line = f.readline()
# 			if not line:
# 				break

# 			if line.startswith(' MONOPOLES'):
# 				while True:
# 					line = f.readline()
# 					tokens = line.split()
# 					print(total_count)
# 					test_charge_map[total_count][count] = float(tokens[1])
					
# 					count += 1
# 					if count == atom_num+bond_num:
						
# 						count = 0
# 						break

# 			if line.startswith(' DIPOLES'):
# 				while True:
# 					line = f.readline()
# 					tokens = line.split()
# 					for i in range(3):
# 						test_multipole_mat[total_count][count][current_index] = float(tokens[i+1])
# 						current_index += 1
# 					current_index -= 3
# 					count += 1
# 					if count == atom_num+bond_num:
# 						current_index += 3
# 						count = 0
# 						break

# 			if line.startswith(' QUADRUPOLES'):
# 				while True:
# 					line = f.readline()
# 					tokens = line.split()
# 					if len(tokens)>2:
# 						for i in range(4):
# 							test_multipole_mat[total_count][count][current_index] = float(tokens[i+1])
# 							current_index += 1
# 					elif len(tokens)==2:
# 						for i in range(2):
# 							test_multipole_mat[total_count][count][current_index] = float(tokens[i])
# 							current_index += 1
# 						current_index -= 6
# 						count += 1
# 					if count == atom_num+bond_num:
# 						current_index += 6
# 						count = 0
# 						break


# 			if line.startswith(' OCTUPOLES'):
# 				while True:
# 					line=f.readline()
# 					tokens = line.split()
# 					if len(tokens)>5:
# 						for i in range(4):
# 							test_multipole_mat[total_count][count][current_index] = float(tokens[i+1])
# 							current_index += 1
# 					elif len(tokens)==5:
# 						for i in range(4):
# 							test_multipole_mat[total_count][count][current_index] = float(tokens[i])
# 							current_index += 1
# 					elif len(tokens)<5:
# 						for i in range(2):
# 							test_multipole_mat[total_count][count][current_index] = float(tokens[i])
# 							current_index += 1
# 						current_index -= 10
# 						count += 1
# 					if count == atom_num+bond_num:
# 						current_index = 0
# 						count = 0
# 						break

			
# 		total_count += 1


# test_charge_map = test_charge_map / charge_sum		
# new_test_multipole_mat = np.zeros((156,19*(bond_num+atom_num)))
# total_count = 0
# for i in range(19):
# 	count = 0
# 	for j in range(bond_num+atom_num):
# 		for k in range(19):
# 			new_test_multipole_mat[total_count][count] = test_multipole_mat[total_count][j][k]
# 			count+=1
# 	total_count+=1


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
	

names = []
test_input_list = np.zeros((19,inp_no))
total_count = 0
count=0
for filename in os.listdir('./yb_test/'):
	if not filename.endswith('.efp'):
		continue;
	with open('./yb_test/' + filename,'r') as f:
		# feat = CoulombMatrix(input_type='filename')
		# feat.fit(['./test/'+filename+'.xyz'])
		# trans = feat.transform(['./test/'+filename+'.xyz'])
		# for i in range(inp_no):
		# 	test_input_list[total_count][i] = trans[0][i]
		names.append(filename)
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


		xyz = open('./yb_test/'+filename+'.xyz', 'r')
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
		


# x_standardscaler = preprocessing.StandardScaler().fit(input_list)
# x_train_scaled = x_standardscaler.transform(input_list)
x_minmaxscaler = preprocessing.MinMaxScaler((-1,1)).fit(input_list)
x_train_scaled = x_minmaxscaler.transform(input_list)


# y_standardscaler = preprocessing.StandardScaler().fit(new_multipole_mat)
# y_train_scaled = y_standardscaler.transform(new_multipole_mat)
# y_minmaxscaler = preprocessing.MinMaxScaler((-1,1)).fit(new_multipole_mat)
# y_train_scaled = y_minmaxscaler.transform(new_multipole_mat)

#joblib.dump(x_standardscaler,'./Trained/inp_std.pkl')
joblib.dump(x_minmaxscaler,'./Trained/inp_minmax.pkl')

# joblib.dump(y_standardscaler,'./Trained/multipole_std.pkl')
# joblib.dump(y_minmaxscaler,'./Trained/multipole_minmax.pkl')


# inputs = Input(shape=(inp_no,))
# # hidden_layer = GaussianDropout(0.2)(inputs)
# hidden_layer = Dense(300,activation='tanh')(inputs)
# hidden_layer = GaussianDropout(0.2)(hidden_layer)
# hidden_layer = Dense(300,activation='tanh')(hidden_layer)
# hidden_layer = GaussianDropout(0.2)(hidden_layer)
# hidden_layer = Dense(300,activation='tanh')(hidden_layer)
# hidden_layer = GaussianDropout(0.2)(hidden_layer)
# hidden_layer = Dense(300,activation='tanh')(hidden_layer)




# output = Dense(19*(bond_num+atom_num), activation='linear')(hidden_layer)
# #output2 = Dense(atom_num+bond_num, activation='softmax')(hidden_layer)
# adam = Adam(lr=0.01)
# model = Model(input=inputs, output=output)
# model.compile(optimizer='nadam', loss='mean_absolute_error')
# model.fit(x_train_scaled, y_train_scaled,epochs=1, batch_size=64,verbose=1,shuffle=True,validation_split=0.1)


# # model.save('./Trained/multipole.h5')

model = KNeighborsRegressor(n_neighbors = 3, weights = 'distance' , p = 1)
model.fit(input_list, charge_map)
joblib.dump(model, 'Trained/charge.pkl')

model = KNeighborsRegressor(n_neighbors = 3, weights = 'distance' , p = 1)
model.fit(input_list, new_multipole_mat)
joblib.dump(model, 'Trained/multipole.pkl')


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
	

# out = open('./multipole.out','w')
# out2 = open('./yb_lysine.out','w')
# for i in range(0,19):
# 	x = test_input_list[i]
# 	#x = x_standardscaler.transform(x.reshape(1, -1))
# 	#x = x_minmaxscaler.transform(x.reshape(1, -1))


# 	start = time.time()
# 	y1 = model.predict(x.reshape(-1,inp_no))


# 	print(time.time()-start)
# 	dist, ind = model.kneighbors(x.reshape(-1,inp_no))
# 	print(names[i], file = out2)
# 	print(dist, file = out2)
# 	print(' ', file = out2)
# 	print(samples_names[ind[0][0]],samples_names[ind[0][1]],samples_names[ind[0][2]], file = out2)
# 	print('---------------------------------------------- ', file = out2)


	#y1 = y_standardscaler.inverse_transform(y1.reshape(1, -1))
	# y1 = y_standardscaler.inverse_transform(y1)
	# print(names[i], file = out)
	# print('Predicted', file=out)
	# print_array_4(y1[0],out)
	# # print('charges:',file=out)
	# # print_array_4(y2[0],out)
	
	# print('Desired', file=out)
	# print_array_4(new_test_multipole_mat[i],out)
	# print('charges:',file=out)
	# print_array_4(test_charge_map[i],out)

	#print('---------------------------------------------',file=out)













