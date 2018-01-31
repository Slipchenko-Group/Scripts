from __future__ import print_function
import sys

with open('./mc_run_filler','r') as mc:
	out = open('mc_run.inp','w')
	while True:
		line = mc.readline()
		if not line:
			break
		if line.startswith(' $F1REF'):
			print(' $F1REF',file=out)
			with open(sys.argv[1],'r') as efp:
				efp.readline()
				while True:
					efp_line = efp.readline()
					if not efp_line:
						
						break
					print(efp_line,end='',file=out)
			continue
		print(line,end='',file=out)		

