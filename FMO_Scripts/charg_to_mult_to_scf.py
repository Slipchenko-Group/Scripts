# Script generates the arrays representing the SCF type 
# and multiplicities of each fragment based on the array that 
# represents the charges of each fragment. 

f1 = open('charg.txt', 'r')
f2 = open('charg.txt.tmp', 'w')
for line in f1:
    f2.write(line.replace('-1', '3'))
f1.close()
f2.close()
f1 = open('charg.txt.tmp', 'r')
f2 = open('charg.txt.tmp2', 'w')
for line in f1:
    f2.write(line.replace('1', '3'))
f1.close()
f2.close()
f1 = open('charg.txt.tmp2', 'r')
f2 = open('charg.txt.tmp', 'w')
for line in f1:
    f2.write(line.replace('-2', '3'))
f1.close()
f2.close()
f1 = open('charg.txt.tmp', 'r')
f2 = open('charg.txt.tmp2', 'w')
for line in f1:
    f2.write(line.replace('2', '3'))
f1.close()
f2.close()
f1 = open('charg.txt.tmp2', 'r')
f2 = open('charg.txt.tmp', 'w')
for line in f1:
    f2.write(line.replace('0', '1'))
f1.close()
f2.close()
f1 = open('charg.txt.tmp', 'r')
f2 = open('mult.txt', 'w')
for line in f1:
    f2.write(line.replace('ICHARG(3)', 'MULT(1)'))
f1.close()
f2.close()
f1 = open('mult.txt', 'r')
f2 = open('scf.txt.tmp', 'w')
for line in f1:
    f2.write(line.replace('3', 'UHF'))
f1.close()
f2.close()
f1 = open('scf.txt.tmp', 'r')
f2 = open('scf.txt.tmp2', 'w')
for line in f1:
    f2.write(line.replace('1', 'RHF'))
f1.close()
f2.close()
f1 = open('scf.txt.tmp2', 'r')
f2 = open('scf.txt', 'w')
for line in f1:
    f2.write(line.replace('MULT(RHF)', 'SCFFRG(1)'))
f1.close()
f2.close()
