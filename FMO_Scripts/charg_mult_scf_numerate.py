# First part of the script generates the arrays representing 
# the SCF type and multiplicities of each fragment based on the array that 
# represents the charges of each fragment. 

import os
import tempfile

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

os.remove('charg.txt.tmp2')
os.remove('charg.txt.tmp')
os.remove('scf.txt.tmp2')
os.remove('scf.txt.tmp')

# Second part of the script numerates nonzero fragments

a1 = open('charg.txt', 'r')
a2 = open('charg_list.txt', 'w')
for line in a1:
    a2.write(line.replace('ICHARG(1)=', ' '))
a1.close()
a2.close()
b1 = open('charg_list.txt', 'r')
b2 = open('charg_formated.txt', 'w')
for line in b1:
    b2.write(line.replace(',', ' '))
b1.close()
b2.close()
arr = []
inp = open ("charg_formated.txt","r")
#read line into array 
for line in inp.readlines():
    # loop over the elemets, split by whitespace
    for i in line.split():
        # convert to integer and append to the list
        arr.append(i)
f4 = open('fragments_listing.txt', 'w')
f3 = list(enumerate(arr))
print >> f4, f3

f5 = open('positive_charged_fragments_id.txt', 'w')
for i, j in f3:
    if j == '1':
     print >> f5, i+1
     
f6 = open('negative_charged_fragments_id.txt', 'w')
for i, j in f3:
    if j == '-1':
     print >> f6, i+1
     
f4.close()
f5.close()
f6.close()

os.remove('charg_list.txt')
os.remove('charg_formated.txt')
os.remove('fragments_listing.txt')
