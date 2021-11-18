#!/usr/bin/python

"""
USAGE: ./create_efrags.py <pdbfile.pdb> <map_file.map>

Credits: Pradeep Gurunathan, Lyudmila V. Slipchenko, Purdue University.

This script makes use of pdb file and map file to generate EFP potentials for individual residues.

BASED ON THE ASSUMPTION THAT THE COLUMN NUMBERS ARE UNMODIFIED, AND SAME AS BROOKHAVEN PDB FILE.

"""
import sys, re, os
from atom_num import mydef1
from math import sqrt


def process_efp(resinfo,file1,file2):
	
	for i in range(len(resinfo)):
		ch1, ch2, nh = '', '', ''
		for j in range(len(resinfo[i])):
			if resinfo[i][j].startswith('name'):
				name = resinfo[i][j].split('=')
				name = name[1]
				name = name.replace(' ','')
			elif resinfo[i][j].startswith('preatoms'):
				preatoms = resinfo[i][j].split('=')
				preatoms = preatoms[1]
				preatoms = preatoms.replace(' ','')
			elif resinfo[i][j].startswith('ch1'):
				ch1 = resinfo[i][j].split('=')
				ch1 = ch1[1]
				ch1 = ch1.replace(' ','')
			elif resinfo[i][j].startswith('ch2'):
				ch2 = resinfo[i][j].split('=')
				ch2 = ch2[1]
				ch2 = ch2.replace(' ','')
			elif resinfo[i][j].startswith('nh'):
				nh = resinfo[i][j].split('=')
				nh = nh[1]
				nh = nh.replace(' ','')
			elif resinfo[i][j].startswith('postatoms'):
				postatoms = resinfo[i][j].split('=')
				postatoms = postatoms[1]
				postatoms = postatoms.replace(' ','')
			elif resinfo[i][j].startswith('rescharge'):
				rescharge = resinfo[i][j].split('=')
				rescharge = rescharge[1]
				rescharge = rescharge.replace(' ','')
			elif resinfo[i][j].startswith('usefp'):
				usefp = resinfo[i][j].split('=')
				usefp = usefp[1]
				usefp = usefp.replace(' ','')
		write_efp(file1, name, preatoms, ch1, ch2, nh, postatoms, rescharge, usefp)
		del name, preatoms, ch1, ch2, nh, postatoms, rescharge, usefp

def write_efp(file1, name, preatoms, ch1, ch2, nh, postatoms, rescharge, usefp):
	postatoms = postatoms.split('-')
	name = name+'.inp'
	try:
		os.remove(name)
	except OSError:
		pass
	inpfile = open(name, 'a')
	inpfile.write(' $contrl units=angs local=boys runtyp=makefp \n')
	inpfile.write('       mult=1 icharg='+str(rescharge)+' coord=cart icut=11 $end\n')
	inpfile.write(' $system timlim=99999   mwords=200 $end\n')
	inpfile.write(' $scf soscf=.f. dirscf=.t. diis=.t. CONV=1.0d-06  $end\n')
	inpfile.write(' $basis gbasis=n31 ngauss=6 ndfunc=1 $end\n')
	inpfile.write(' $DAMP IFTTYP(1)=2,0 IFTFIX(1)=1,1 thrsh=500.0 $end\n')
	inpfile.write(' $MAKEFP  POL=.t. DISP=.t. CHTR=.f.  EXREP=.t. $end\n $data\n')
	inpfile.write(' '+str(usefp))
	inpfile.write('\n C1\n')
	for i in range(int(postatoms[0]),int(postatoms[1])+1):
		for j in range(len(file1)):
			if int(file1[j][6:11]) == i:
				inpfile.write(str(' ')+str(file1[j][76:78].replace(" ",""))+str(file1[j][6:11]).replace(' ',''))
				inpfile.write(str('   ')+str(mydef1(str(file1[j][76:78]))))
				inpfile.write(str('.0  ')+str(file1[j][30:54])+str('\n'))
				break
	if ch1 != '':
		ch1 = ch1.split(',')
		x3,y3,z3 = add_hydrogens(file1,'ch',ch1)
		inpfile.write(str(' H000    1.0    ')+str(x3)+str('  ')+str(y3)+str('  ')+str(z3)+str(' \n'))
	if ch2 != '':
		ch2 = ch2.split(',')
		x3,y3,z3 = add_hydrogens(file1,'ch',ch2)
		inpfile.write(str(' H000    1.0    ')+str(x3)+str('  ')+str(y3)+str('  ')+str(z3)+str(' \n'))
	if nh != '':
		nh = nh.split(',')
		x3,y3,z3 = add_hydrogens(file1,'nh',nh)
		inpfile.write(str(' H0000   1.0    ')+str(x3)+str('  ')+str(y3)+str('  ')+str(z3)+str(' \n'))
	inpfile.write(' $end \n')
	inpfile.write(' $comment Atoms to be erased: ')						# Redundant: Not required. Can comment it out...
	inpfile.write(' $end\n\n')
	inpfile.close()


def add_hydrogens(file1,h_type,start_end):

	if h_type == 'ch':
		desired_length = 1.07886
	if h_type == 'nh':
		desired_length = 1.00339
	start_end[0] = int(start_end[0])
	start_end[1] = int(start_end[1])
	for j in range(len(file1)):
		if int(file1[j][6:11]) == start_end[0]:
			x1 = float(file1[j][30:38])
			y1 = float(file1[j][38:46])
			z1 = float(file1[j][46:54])
			break
	for j in range(len(file1)):
		if int(file1[j][6:11]) == start_end[1]:
			x2 = float(file1[j][30:38])
			y2 = float(file1[j][38:46])
			z2 = float(file1[j][46:54])
			break
	actual_length = sqrt(((x1-x2)**2)+((y1-y2)**2)+((z1-z2)**2))
	x3 = ((x2-x1)*desired_length/actual_length)+x1
	y3 = ((y2-y1)*desired_length/actual_length)+y1
	z3 = ((z2-z1)*desired_length/actual_length)+z1
	return (x3,y3,z3)

def main(pdbfile,mapfile):
	try:
		inputPDB=open(pdbfile,'r')												# Try opening the file
		inputmap=open(mapfile,'r')												# Try opening the file
	except IOError:
		sys.stderr.write('Failed to open the PDB and/or MAP file')				# Throw error if non-existent
		return 0
	
	lines = inputPDB.readlines()
	inputPDB.close()
	
	inputPDB=open(pdbfile,'w')
	
	for line in lines:
		if line.startswith('ATOM') or line.startswith('HETATM'):
			inputPDB.write(line)
	
	inputPDB.close()
	
	inputPDB=open(pdbfile,'r')													# Opening the PDB file again
	
	file1 = inputPDB.read()														# Read lines and split by newline character												
	file1 = file1.split('\n')													# Read lines and split by newline character												
	file2 = inputmap.read()														# Read lines and split by newline character
	file2 = file2.split('\n')													# Read lines and split by newline character

	
	numres = file2.count('$residue')											# Number of $residue instances
	resinfo = []
	resnum = -1																	# Initialize residue number count
	for i in range(len(file2)):
		if file2[i].startswith('$residue'):
			resnum = resnum+1
			resinfo.append([])
		elif file2[i].endswith('$end ') or file2[i] == '':
			pass
		else:
			resinfo[resnum].append(file2[i])
	
	process_efp(resinfo, file1, file2)
	
	
main(sys.argv[1],sys.argv[2])
