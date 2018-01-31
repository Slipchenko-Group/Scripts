from __future__ import print_function
import os

no = 0
for name in os.listdir('./ref'):
	if name.startswith('ref_'):
		no += 1

ref = open('ref/bash.sh','w')
sat = open('sat/bash.sh','w')

current = 0
for i in range(19):
	print('for i in ref_ss{'+str(current)+'..'+str(current+no/20)+'}.inp ; do /group/lslipche/apps/libefp/libefp_09012017/libefp/bin/efpmd $i > ./output/$i.log ; done &',file=ref)
	print('for i in sat_ss{'+str(current)+'..'+str(current+no/20)+'}.inp ; do /group/lslipche/apps/libefp/libefp_09012017/libefp/bin/efpmd $i > ./output/$i.log ; done &',file=sat)
	current = current+no/20+1

print('for i in ref_ss{'+str(current)+'..'+str(no-1)+'}.inp ; do /group/lslipche/apps/libefp/libefp_09012017/libefp/bin/efpmd $i > ./output/$i.log ; done',file=ref)
print('for i in sat_ss{'+str(current)+'..'+str(no-1)+'}.inp ; do /group/lslipche/apps/libefp/libefp_09012017/libefp/bin/efpmd $i > ./output/$i.log ; done',file=sat)	

ref.close()
sat.close()
