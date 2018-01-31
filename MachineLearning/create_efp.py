from __future__ import print_function
import numpy as np
import os
import itertools
from detect_bonds import Detect_bonds
from sklearn import preprocessing
from sklearn.externals import joblib
import pickle
import time
from molml.features import CoulombMatrix

charge_model = joblib.load('Trained/charge.pkl')
multipole_model = joblib.load('Trained/multipole.pkl')
lmp_geo_model = joblib.load('Trained/lmo_geo.pkl')
static_pol_model = joblib.load('Trained/static_pol.pkl')
dynamic_pol_model = joblib.load('Trained/dynamic_pol.pkl')
wave_model = joblib.load('Trained/wave.pkl')
fock_model = joblib.load('Trained/fock.pkl')
screen1_model = joblib.load('Trained/screen1.pkl')
screen2_model = joblib.load('Trained/screen2.pkl')



inp_minmax = joblib.load('Trained/inp_minmax.pkl')
# multipole_std = joblib.load('Trained/multipole_std.pkl')
# multipole_minmax = joblib.load('Trained/multipole_minmax.pkl')
# pol_std = joblib.load('Trained/pol_std.pkl')
# pol_minmax = joblib.load('Trained/pol_minmax.pkl')
# pol_geo_std = joblib.load('Trained/pol_geo_std.pkl')
# pol_geo_minmax = joblib.load('Trained/pol_geo_minmax.pkl')
# dispersion_std = joblib.load('Trained/dispersion_std.pkl')
# dispersion_minmax = joblib.load('Trained/dispersion_minmax.pkl')
# wave_std = joblib.load('Trained/wave_std.pkl')
# wave_minmax = joblib.load('Trained/wave_minmax.pkl')
# fock_std = joblib.load('Trained/fock_std.pkl')
# fock_minmax = joblib.load('Trained/fock_minmax.pkl')
# screen1_std = joblib.load('Trained/screen1_std.pkl')
# screen1_minmax = joblib.load('Trained/screen1_minmax.pkl')
# screen2_std = joblib.load('Trained/screen2_std.pkl')
# screen2_minmax = joblib.load('Trained/screen2_minmax.pkl')


with open('./ref/ref','rb') as ref_data:
	metadata = pickle.load(ref_data)
atom_num = metadata['atom_num']
bond_num = metadata['bond_num']
lmo_count = metadata['lmo_count']
bonds = metadata['bonds']
ref_geo = metadata['ref_geo']
ref_lmo = metadata['ref_lmo']
ref_lmo_log = metadata['ref_lmo_log']
inp_no = metadata['input_no']
#inp_no = atom_num * atom_num
charge_sum = metadata['charge_sum']
moments = metadata['moments']
charge_suffix = metadata['charge_suffix']
coord_suffix = metadata['coord_suffix']
dynamic_pol_suffix = metadata['dynamic_pol_suffix']
projection_basis_set = metadata['projection_basis_set']
multiplicity = metadata['multiplicity']
wave_per_lmo = metadata['wave_per_lmo']
fragment_name = raw_input('fragment name (case sensitive): ')

distance = open('distance.out','w')

while True:
	fname = raw_input('path to xyz file: ')
	if not fname.endswith('.xyz'):
		continue
	start_time = time.time()
	K, ext = os.path.splitext(os.path.basename(fname))
	out_file = open('efp/'+K[:-4]+'_saturated.efp','w')
	try:
		db = Detect_bonds(fname)
	except:
		print("Wrong file path, please input again...")
		continue
	bond_len = db.all_lengths(bonds)
	bond_ang = db.all_bond_angles(bonds)
	dih_ang = db.all_dihedral_angles(bonds)
	mid = db.get_medium_coord(bonds)
	cartesian = db.get_cartesian()
	# print(' $FRAGNAME', file=out_file)
	print(' $'+fragment_name,file=out_file)
	print('EFP DATA FOR FRAGNAME SCFTYP=RHF     ... GENERATED WITH BASIS SET=XXX',file=out_file)
	print(' COORDINATES (BOHR)',file=out_file)
	for i in range(atom_num):
		print(moments[i]+'      '+str(format(float(cartesian[i][0]),' .10f'))+'  '+str(format(float(cartesian[i][1]),' .10f'))+'  '+str(format(float(cartesian[i][2]),' .10f'))
			+'  '+coord_suffix[i]+'  '+charge_suffix[i],file=out_file)
	for i in range(bond_num):
		print(moments[i+atom_num]+'      '+str(format(float(mid[i][0]),' .10f'))+'  '+str(format(float(mid[i][1]),' .10f'))+'  '+str(format(float(mid[i][2]),' .10f'))
			+'  '+coord_suffix[i+atom_num]+'  '+charge_suffix[i+atom_num],file=out_file)
	print(' STOP',file=out_file)

	inp = np.zeros(inp_no)
	count = 0
	for i in range(len(bond_len)):
		inp[count] = bond_len[i]
		count+=1
	for i in range(len(bond_ang)):
		inp[count] = bond_ang[i]
		count+=1
	for i in range(len(dih_ang)):
		inp[count] = dih_ang[i]
		count+=1
	

	inp = inp_minmax.transform(inp.reshape(1,-1))
	charges = charge_model.predict(inp)
	dist, ind = multipole_model.kneighbors(inp)
	print(fname, file = distance)
	print(dist, file = distance)
	print('---------------------------------------------- ', file = distance)
	
	multipoles = multipole_model.predict(inp)
	# multipoles,charges = multipole.predict(inp)
	# multipoles = multipole_minmax.inverse_transform(multipoles.reshape(1,-1))
	# multipoles = multipole_std.inverse_transform(multipoles)
	print(' MONOPOLES',file=out_file)
	for i in range(atom_num+bond_num):
		print(moments[i]+'      '+str(format(charges[0][i]*charge_sum,' .10f')+'   '+charge_suffix[i]),file=out_file)
	print(' STOP',file=out_file)
	print(' DIPOLES',file=out_file)
	for i in range(atom_num+bond_num):
		print(moments[i]+'       '+str(format(multipoles[0][19*i],' .10f'))+'    '+
			str(format(multipoles[0][19*i+1],' .10f'))+'    '+
			str(format(multipoles[0][19*i+2],' .10f')),file=out_file)
	print(' STOP',file=out_file)
	print(' QUADRUPOLES',file=out_file)
	for i in range(atom_num+bond_num):
		print(moments[i]+'       '+str(format(multipoles[0][19*i+3],' .10f'))+'   '+
			str(format(multipoles[0][19*i+4],' .10f'))+'   '+
			str(format(multipoles[0][19*i+5],' .10f')) + '   '+
			str(format(multipoles[0][19*i+6],' .10f')) + ' >',file=out_file)
		print('           '+str(format(multipoles[0][19*i+7],' .10f'))+'   '+
			str(format(multipoles[0][19*i+8],' .10f')),file=out_file)
	print(' STOP',file=out_file)
	print(' OCTUPOLES',file=out_file)
	for i in range(atom_num+bond_num):
		print(moments[i]+'         '+str(format(multipoles[0][19*i+9],' .9f'))+'   '+
			str(format(multipoles[0][19*i+10],' .9f'))+'   '+
			str(format(multipoles[0][19*i+11],' .9f')) + '   '+
			str(format(multipoles[0][19*i+12],' .9f')) + ' >',file=out_file)
		print('             '+str(format(multipoles[0][19*i+13],' .9f'))+'   '+
			str(format(multipoles[0][19*i+14],' .9f'))+'   '+
			str(format(multipoles[0][19*i+15],' .9f')) + '   '+
			str(format(multipoles[0][19*i+16],' .9f')) + ' >',file=out_file)
		print('             '+str(format(multipoles[0][19*i+17],' .9f'))+'   '+
			str(format(multipoles[0][19*i+18],' .9f')),file=out_file)
	print(' STOP',file=out_file)
	lmo_geo = lmp_geo_model.predict(inp)
	pols = static_pol_model.predict(inp)

	# lmo_geo, pols = pol.predict(inp)
	# lmo_geo = pol_geo_minmax.inverse_transform(lmo_geo.reshape(1,-1))
	# lmo_geo = pol_geo_std.inverse_transform(lmo_geo)
	# pols = pol_minmax.inverse_transform(pols.reshape(1,-1))
	# pols = pol_std.inverse_transform(pols)


	print(' POLARIZABLE POINTS',file=out_file)
	for i in range(lmo_count):
		print('CT'+str(i+1)+'  '+str(format(lmo_geo[0][i*3],' .10f'))+'  '+str(format(lmo_geo[0][i*3+1],' .10f'))
			+'  '+str(format(lmo_geo[0][i*3+2],' .10f')),file=out_file)
		print('   '+str(format(pols[0][i*9],' .10f'))+'   '+str(format(pols[0][i*9+1],' .10f'))+'   '+
			str(format(pols[0][i*9+2],' .10f'))+'   '+str(format(pols[0][i*9+3],' .10f'))
			+' >',file=out_file)
		print('   '+str(format(pols[0][i*9+4],' .10f'))+'   '+str(format(pols[0][i*9+5],' .10f'))+'   '+
			str(format(pols[0][i*9+6],' .10f'))+'   '+str(format(pols[0][i*9+7],' .10f'))
			+' >',file=out_file)
		print('   '+str(format(pols[0][i*9+8],' .10f')),file=out_file)
	print(' STOP',file=out_file)

	# dynamic_pol = dispersion.predict(inp)
	# dynamic_pol = dispersion_minmax.inverse_transform(dynamic_pol.reshape(1,-1))
	# dynamic_pol = dispersion_std.inverse_transform(dynamic_pol)
	dynamic_pol = dynamic_pol_model.predict(inp)

	print(' DYNAMIC POLARIZABLE POINTS',file=out_file)
	for i in range(12):
		for j in range(lmo_count):
			if j == 0:
				print('CT '+str(format(j+1,'2d'))+'  '+str(format(lmo_geo[0][j*3], ' .10f'))+
					'  '+str(format(lmo_geo[0][j*3+1],' .10f')) + '  '+
					str(format(lmo_geo[0][j*3+2],' .10f'))+dynamic_pol_suffix[i],file=out_file)
			else:
				print('CT '+str(format(j+1,'2d'))+'  '+str(format(lmo_geo[0][j*3], ' .10f'))+
					'  '+str(format(lmo_geo[0][j*3+1],' .10f')) + '  '+
					str(format(lmo_geo[0][j*3+2],' .10f')),file=out_file)

			print('   '+str(format(dynamic_pol[0][i*lmo_count*9+j*9],' .10f'))+'   '+
				str(format(dynamic_pol[0][i*lmo_count*9+j*9+1],' .10f'))+'   '+
				str(format(dynamic_pol[0][i*lmo_count*9+j*9+2],' .10f'))+'   '+
				str(format(dynamic_pol[0][i*lmo_count*9+j*9+3],' .10f'))+' >',file=out_file)
			print('   '+str(format(dynamic_pol[0][i*lmo_count*9+j*9+4],' .10f'))+'   '+
				str(format(dynamic_pol[0][i*lmo_count*9+j*9+5],' .10f'))+'   '+
				str(format(dynamic_pol[0][i*lmo_count*9+j*9+6],' .10f'))+'   '+
				str(format(dynamic_pol[0][i*lmo_count*9+j*9+7],' .10f'))+' >',file=out_file)
			print('   '+str(format(dynamic_pol[0][i*lmo_count*9+j*9+8],' .10f')),file=out_file)
	print(' STOP',file=out_file)
	print(' PROJECTION BASIS SET',file=out_file)
	for i in range(atom_num):
		print(moments[i]+'        '+str(format(cartesian[i][0],' .10f'))
			+'  '+str(format(cartesian[i][1],' .10f'))+'  '+
			str(format(cartesian[i][2],' .10f'))+'    '+projection_basis_set[i][0],file=out_file)
		for j in range(1,len(projection_basis_set[i])):
			print((projection_basis_set[i][j]).replace('\n',''),file=out_file)
		print('  ',file=out_file)
	print(' STOP',file=out_file)
	print(' MULTIPLICITY    '+str(multiplicity).replace('\n',''),file=out_file)
	print(' STOP',file=out_file);
	print(' PROJECTION WAVEFUNCTION     '+str(lmo_count)+'    '+str(format(wave_per_lmo,'3d')),file=out_file)
	# wave_functions, fock = repulsion.predict(inp)
	# wave_functions = wave_minmax.inverse_transform(wave_functions.reshape(1,-1))
	# wave_functions = wave_std.inverse_transform(wave_functions)

	wave_functions = wave_model.predict(inp)
	fock = fock_model.predict(inp)

	# fock = fock_minmax.inverse_transform(fock.reshape(1,-1))
	# fock = fock_std.inverse_transform(fock)
	lines_of_5 = wave_per_lmo / 5
	num_last_line = wave_per_lmo % 5

	for i in range(lmo_count):
		for j in range(lines_of_5):
			print(str(format(i+1,'2d'))+str(format(j+1,'3d'))+str(format(wave_functions[0][wave_per_lmo*i+5*j],' .8E'))+
				str(format(wave_functions[0][wave_per_lmo*i+5*j+1],' .8E'))+
				str(format(wave_functions[0][wave_per_lmo*i+5*j+2],' .8E'))+
				str(format(wave_functions[0][wave_per_lmo*i+5*j+3],' .8E'))+
				str(format(wave_functions[0][wave_per_lmo*i+5*j+4],' .8E')), file=out_file)
		if num_last_line == 1:
			print(str(format(i+1,'2d'))+str(format(lines_of_5+1,'3d'))+str(format(wave_functions[0][wave_per_lmo*i+5*lines_of_5],' .8E')),file=out_file)
		elif num_last_line ==2:
			print(str(format(i+1,'2d'))+str(format(lines_of_5+1,'3d'))+str(format(wave_functions[0][wave_per_lmo*i+5*lines_of_5],' .8E'))+
				str(format(wave_functions[0][wave_per_lmo*i+5*lines_of_5+1],' .8E')),file=out_file)
		elif num_last_line ==3:
			print(str(format(i+1,'2d'))+str(format(lines_of_5+1,'3d'))+str(format(wave_functions[0][wave_per_lmo*i+5*lines_of_5],' .8E'))+
				str(format(wave_functions[0][wave_per_lmo*i+5*lines_of_5+1],' .8E'))+
				str(format(wave_functions[0][wave_per_lmo*i+5*lines_of_5+2],' .8E')),file=out_file)
		elif num_last_line ==4:
			print(str(format(i+1,'2d'))+str(format(lines_of_5+1,'3d'))+str(format(wave_functions[0][wave_per_lmo*i+5*lines_of_5],' .8E'))+
				str(format(wave_functions[0][wave_per_lmo*i+5*lines_of_5+1],' .8E'))+
				str(format(wave_functions[0][wave_per_lmo*i+5*lines_of_5+2],' .8E'))+
				str(format(wave_functions[0][wave_per_lmo*i+5*lines_of_5+3],' .8E')),file=out_file)
	print(' FOCK MATRIX ELEMENTS',file=out_file)
	line_of_4 = (len(fock[0])) / 4
	num_last_line = (len(fock[0])) % 4
	for i in range(line_of_4):
		print('   '+str(format(fock[0][i*4],' .10f'))+'   '+str(format(fock[0][i*4+1],' .10f'))+'   '+
			str(format(fock[0][i*4+2],' .10f'))+'   '+str(format(fock[0][i*4+3],' .10f'))+' >',file=out_file)
	if num_last_line == 1:
		print('   '+str(format(fock[0][line_of_4*4],' .10f')),file=out_file)
	elif num_last_line ==2:
		print('   '+str(format(fock[0][line_of_4*4],' .10f'))+'   '+str(format(fock[0][line_of_4*4+1],' .10f')),file=out_file)
	elif num_last_line ==3:
		rint('   '+str(format(fock[0][line_of_4*4],' .10f'))+'   '+str(format(fock[0][line_of_4*4+1],' .10f'))+'   '+
			str(format(fock[0][line_of_4*4+2],' .10f')),file=out_file)
	print(' LMO CENTROIDS',file=out_file)
	for i in range(lmo_count):
		print('CT'+str(i+1)+'  '+str(format(lmo_geo[0][i*3],' .10f'))+'  '+
			str(format(lmo_geo[0][i*3+1],' .10f'))+'  '+
			str(format(lmo_geo[0][i*3+2],' .10f')),file=out_file)
	print(' STOP',file=out_file)
	print('SCREEN2      (FROM VDWSCL=   0.700)',file=out_file)

	screen1 = screen1_model.predict(inp)
	screen2 = screen2_model.predict(inp)
	# screen1, screen2 = screen.predict(inp)
	# screen1 = screen1_minmax.inverse_transform(screen1.reshape(1,-1))
	# screen1 = screen1_std.inverse_transform(screen1)
	# screen2 = screen2_minmax.inverse_transform(screen2.reshape(1,-1))
	# screen2 = screen2_std.inverse_transform(screen2)
	for i in range(len(moments)):
		if screen2[0][i] > 10:
			screen2[0][i] = 10
		elif screen2[0][i] < 0:
			screen2[0][i] = 0
		print(' '+moments[i]+'       1.000000000  '+str(format(screen2[0][i],'.9f')),file=out_file)
	print('STOP',file=out_file)
	print('SCREEN       (FROM VDWSCL=   0.700)',file=out_file)
	for i in range(len(moments)):
		if screen1[0][i] > 10:
			screen1[0][i] = 10
		elif screen1[0][i] < 0:
			screen1[0][i] = 0
		print(' '+moments[i]+'       1.000000000  '+str(format(screen1[0][i],'.9f')),file=out_file)
	print('STOP',file=out_file)
	print(' $END',file=out_file)

	out_file.close()
	print(time.time()-start_time)



