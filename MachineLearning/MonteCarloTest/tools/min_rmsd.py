from __future__ import print_function
import calculate_rmsd as rmsd
import os
import sys

all_rmsd = []
q_atoms, Q = rmsd.get_coordinates(sys.argv[1],'xyz')
Q -= rmsd.centroid(Q)

for filename in os.listdir('../../ML/samples3-1-1000'):
	if not filename.endswith('.xyz'):
		continue
	p_atoms, P = rmsd.get_coordinates('../../ML/samples3-1-1000/'+filename,'xyz')
	P -= rmsd.centroid(P)
	all_rmsd.append(rmsd.kabsch_rmsd(P, Q))
file = open('./rmsd_results/min_rmsd.csv','w')
for rmsd in all_rmsd:
	print(rmsd,file=file)
file.close()