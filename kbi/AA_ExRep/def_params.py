import pandas as pd
import numpy as np
import re
from scipy.spatial import distance
import os

def get_params(efp):

	#If the two closest distances to lmo are < 2 then bond.
	#If one of the two closest atoms is more than 2A bohr away then lone pair.	
	#Assumptions all nitrogens are sp3 (for lone pairs) [must fix]
	#Two versions for aromatic carbons due to diff localization scheme:
	    # version 1: treat C-C as reguar C-C and C=C as regular C=C
	    # version 2: treat each C=C pair as two separate aromatic C-C and each C-C as regular C-C

	params_dict = {\
	               'CH1' : [0.54, 0.41],
	               'CC1' : [0.00, 0.00],
	               'CC2' : [0.11, 0.21],
	               'CC3' : [0.18, 0.25],
	               'O0'  : [0.70, 0.50],
	               'HO1' : [0.40, 0.40],
	               'CO1' : [0.00, 0.00],
	               'CO2' : [1.02, 1.16],
	               'N0'  : [0.11, 0.23],
	               'N02' : [0.60, 0.40],
	               'HN1' : [1.60, 0.62],
	               'CN1' : [0.00, 0.00],
	               'CN2' : [0.30, 0.80],
	               'CN3' : [1.90, 0.78],
	               'ACH1' : [0.54, 0.41],
	               'ACC1' : [1.92, 0.45],
	               'ACC2' : [0.00, 0.00]
	              }
	
	efp_lns = open(f'{os.getcwd()}/efparm/{efp}', 'r').readlines()
	
	# find coordinates of atoms
	bohr_line = efp_lns.index(' COORDINATES (BOHR)\n')
	mon_line = efp_lns.index(' MONOPOLES \n')
	atoms = [line.split()[0:-2] for line in \
	         efp_lns[bohr_line:mon_line-1] if 'B' not in line]
	atoms_dict = {}
	for natom in range(0, len(atoms)):
	    atoms[natom][0] = atoms[natom][0].replace(atoms[natom][0],
	                                              re.findall("[a-zA-Z]+", atoms[natom][0][1:])[0])
	for natom in range(0, len(atoms)):
	    atoms_dict[f'{natom+1}'] = atoms[natom][0]
	atoms = [[natom[0], float(natom[1]), float(natom[2]), float(natom[3])] for natom in atoms]
	    
	# find coordinates of lmos
	lmo_line = efp_lns.index(' LMO CENTROIDS\n')
	screen2_line = efp_lns.index('SCREEN2      (FROM VDWSCL=   0.700)\n')
	lmos = [line.split() for line in efp_lns[lmo_line+1:screen2_line-1]]
	lmos = [[nlmo[0], float(nlmo[1]), float(nlmo[2]), float(nlmo[3])] for nlmo in lmos]
	
	# create dataframe
	data_dict = {'lmo' : [], 'A1' : [], 'A1_dist' : [], 'A2' : [], 'A2_dist' : []}
	for lmo in lmos:
	    distances = []
	    atoms_ids = []
	    dists = {}
	    for natom in range(0, len(atoms)):
	        atom = atoms[natom]
	        key = natom+1
	        dist = distance.euclidean(lmo[1:], atom[1:])
	        atoms_ids.append(key)
	        distances.append(dist)
	        dists[f'{key}'] = dist
	    min_dists = list(dists.values())
	    min_dists.sort()
	    min_dists = min_dists[0:2]
	    A1 = [atom for atom in dists.keys() if dists[atom] == min_dists[0]][0]
	    A2 = [atom for atom in dists.keys() if dists[atom] == min_dists[1]][0]
	    data_dict['lmo'].append(lmo[0])
	    data_dict['A1'].append(int(A1))
	    data_dict['A1_dist'].append(min_dists[0])
	    data_dict['A2'].append(int(A2))
	    data_dict['A2_dist'].append(min_dists[1])
	
	sample = pd.DataFrame(data_dict)[['lmo', 'A1', 'A2', 'A1_dist', 'A2_dist']]
	sample.loc[:,'id'] = sample.index
	
	bond = lambda x: 1 if x < 2 else 0
	
	sample.loc[:, 'bond'] = sample['A2_dist'].map(bond)
	sample.loc[:,'A1_num'] = sample[['A1', 'A2']].min(axis=1)
	sample.loc[:,'A2_num'] = sample[['A1', 'A2']].max(axis=1)
	
	lps = sample[sample['bond']==0]
	types = sample[sample['bond']==1][['A1_num', 'A2_num', 'bond']]\
	        .groupby(['A1_num', 'A2_num']).agg({'bond' : 'sum'}).reset_index()\
	        .rename(columns={'bond' : 'lmo_type'})
	
	# Lone pairs data frame
	lps = lps.drop('A1_num', axis=1).drop('A2_num', axis=1)
	lps.loc[:,'A1_num'] = lps['A1']
	lps.loc[:,'A2_num'] = 0
	lps = lps[['lmo', 'A1_num', 'A2_num', 'id']]
	lps.loc[:,'lmo_type'] = 0
	
	# bonds data frame
	bonds = sample[sample['bond']==1][['lmo', 'A1_num', 'A2_num', 'id']]\
	        .merge(types, left_on=['A1_num', 'A2_num'], 
	                      right_on=['A1_num', 'A2_num'],
	                      how='left')
	
	lmo_types = pd.concat([bonds, lps])
	lmo_types = lmo_types[['id', 'lmo', 'A1_num', 'A2_num', 'lmo_type']].sort_values('id')
	def get_cat(A1_num, A2_num, lmo_type):
	    if lmo_type == 0:
	        return(atoms_dict[f'{A1_num}']+f'{lmo_type}')
	    else:
	        return("".join(sorted(atoms_dict[f'{A1_num}']+atoms_dict[f'{A2_num}']))+f'{lmo_type}')
	    
	#lmo_types['category'] = lmo_types[['A1_num', 'A2_num']]\
	#                        .apply(lambda x: get_cat(x.A1_num, x.A2_num), axis=1)
	
	lmo_types.loc[:,'category'] = lmo_types[['A1_num', 'A2_num', 'lmo_type']]\
	                        .apply(lambda x: get_cat(x.A1_num, x.A2_num, x.lmo_type), axis=1)
	

	lmo_types.loc[:,'beta_noAr'] = lmo_types['category'].apply(lambda x: params_dict[x][0])
	lmo_types.loc[:,'alpha_noAr'] = lmo_types['category'].apply(lambda x: params_dict[x][1])
	lmo_types.loc[:,'beta_v2Ar'] = lmo_types['category'].apply(lambda x: params_dict[x.replace('CC2', 'ACC1')][0])
	lmo_types.loc[:,'alpha_v2Ar'] = lmo_types['category'].apply(lambda x: params_dict[x.replace('CC2', 'ACC1')][1])

	return(lmo_types)
