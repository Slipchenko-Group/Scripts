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
from sklearn.externals import joblib
import lmo_arrange as la
import pickle
import time
from molml.features import CoulombMatrix
from sklearn.neighbors import KNeighborsRegressor



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
#inp_no = metadata['input_no']
inp_no = atom_num * 3
charge_sum = metadata['charge_sum']
total_mass = metadata['total_mass']
masses = {"C":12.0107, "O":15.199, "N":14.0067, "H":1.00794}
	


screen2 = np.zeros((7757,atom_num+bond_num))
screen1 = np.zeros((7757,atom_num+bond_num))

current_index = 0
num_inp = 0
total_count = 0
count = 0


for filename in os.listdir('samples/'):

	if not filename.endswith('.efp'):
		continue
	with open('./samples/' + filename,'r') as f:
		curr_index = 0
		while True:
			line = f.readline()
			if not line:
				break
			if line.startswith('SCREEN2'):
				curr_index = 0
				while True:
					line = f.readline()
					if line.startswith('STOP'):
						break
					tokens = line.split()
					screen2[total_count][curr_index] = float(tokens[-1])
					curr_index+=1

			if line.startswith('SCREEN'):
				curr_index = 0
				while True:
					line = f.readline()
					if line.startswith('STOP'):
						break
					tokens = line.split()
					screen1[total_count][curr_index] = float(tokens[-1])
					curr_index+=1

		total_count += 1



# test_screen2 = np.zeros((5,atom_num+bond_num))
# test_screen1 = np.zeros((5,atom_num+bond_num))
# current_index = 0
# total_count = 0
# count = 0

# for filename in os.listdir('test3-1-1000/'):
# 	if not filename.endswith('.efp'):
# 		continue
# 	with open('./test3-1-1000/' + filename,'r') as f:

# 		curr_index = 0
# 		while True:
# 			line = f.readline()
# 			if not line:
# 				break
# 			if line.startswith('SCREEN2'):
# 				curr_index = 0
# 				while True:
# 					line = f.readline()
# 					if line.startswith('STOP'):
# 						break
# 					tokens = line.split()
# 					test_screen2[total_count][curr_index] = float(tokens[-1])
# 					curr_index+=1

# 			if line.startswith('SCREEN'):
# 				curr_index = 0
# 				while True:
# 					line = f.readline()
# 					if line.startswith('STOP'):
# 						break
# 					tokens = line.split()
# 					test_screen1[total_count][curr_index] = float(tokens[-1])
# 					curr_index+=1

# 		total_count += 1


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
	
	

# test_input_list = np.zeros((5,inp_no))
# total_count = 0
# count=0
# for filename in os.listdir('./test3-1-1000/'):
# 	if not filename.endswith('.efp'):
# 		continue;
# 	with open('./test3-1-1000/' + filename,'r') as f:

# 		db = Detect_bonds('./test3-1-1000/' + filename)
# 		bond_len = db.all_lengths(bonds)
# 		bond_ang = db.all_bond_angles(bonds)
# 		dih_ang = db.all_dihedral_angles(bonds)

		
# 		loc_count = 0
# 		for i in range(len(bond_len)):
# 			test_input_list[total_count][loc_count] = bond_len[i]
# 			loc_count = loc_count+1
# 		for i in range(len(bond_ang)):
# 			test_input_list[total_count][loc_count] = bond_ang[i]
# 			loc_count = loc_count+1
# 		for i in range(len(dih_ang)):
# 			test_input_list[total_count][loc_count] = dih_ang[i]
# 			loc_count = loc_count+1

# 		total_count = total_count+1
		



# x_minmaxscaler = joblib.load('Trained/inp_minmax.pkl')
# x_train_scaled = x_minmaxscaler.transform(input_list)




# y1_standardscaler = preprocessing.StandardScaler().fit(screen1)
# y1_train_scaled = y1_standardscaler.transform(screen1)
# y1_minmaxscaler = preprocessing.MinMaxScaler((0,1)).fit(y1_train_scaled)
# y1_train_scaled = y1_minmaxscaler.transform(y1_train_scaled)

# y2_standardscaler = preprocessing.StandardScaler().fit(screen2)
# y2_train_scaled = y2_standardscaler.transform(screen2)
# y2_minmaxscaler = preprocessing.MinMaxScaler((0,1)).fit(y2_train_scaled)
# y2_train_scaled = y2_minmaxscaler.transform(y2_train_scaled)





# inputs = Input(shape=(inp_no,))
# hidden_layer = Dense(500,activation='relu')(inputs)
# hidden_layer = GaussianDropout(0.2)(hidden_layer)
# hidden_layer = Dense(500,activation='relu')(hidden_layer)
# hidden_layer = GaussianDropout(0.2)(hidden_layer)
# hidden_layer = Dense(500,activation='relu')(hidden_layer)
# hidden_layer = GaussianDropout(0.2)(hidden_layer)
# hidden_layer = Dense(500,activation='relu')(hidden_layer)
# hidden_layer = GaussianDropout(0.2)(hidden_layer)
# hidden_layer = Dense(500,activation='relu')(hidden_layer)


model = KNeighborsRegressor(n_neighbors = 3, weights = 'distance' , p = 1)
model.fit(input_list, screen1)
joblib.dump(model, 'Trained/screen1.pkl')

model = KNeighborsRegressor(n_neighbors = 3, weights = 'distance' , p = 1)
model.fit(input_list, screen2)
joblib.dump(model, 'Trained/screen2.pkl')


# output = Dense(bond_num+atom_num, activation='sigmoid')(hidden_layer)
# output2 = Dense(atom_num+bond_num, activation='sigmoid')(hidden_layer)
# adam = Adam(lr=0.0005)
# model = Model(input=inputs, output=[output,output2])
# model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mean_absolute_error'])
# model.fit(x_train_scaled, [y1_train_scaled,y2_train_scaled],epochs=500, batch_size=64,verbose=1,shuffle=True,validation_split=0.1)

# model.save('./Trained/screen.h5')

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
	


# for i in range(0,5):
# 	x = test_input_list[i]
# 	x = x_standardscaler.transform(x.reshape(1, -1))
# 	x = x_minmaxscaler.transform(x)
# 	y1,y2 = model.predict(x.reshape(-1,inp_no))

# 	y1 = y1_minmaxscaler.inverse_transform(y1.reshape(1, -1))
# 	y1 = y1_standardscaler.inverse_transform(y1)
# 	y2 = y2_minmaxscaler.inverse_transform(y2.reshape(1, -1))
# 	y2 = y2_standardscaler.inverse_transform(y2)

# 	print('Predicted', file=out)
# 	print_array_4(y1[0],out)
# 	print('charges:',file=out)
# 	print_array_4(y2[0],out)
	
# 	print('Desired', file=out)
# 	print_array_4(test_screen1[i],out)
# 	print('charges:',file=out)
# 	print_array_4(test_screen2[i],out)

# 	print('---------------------------------------------',file=out)

# out.close()











