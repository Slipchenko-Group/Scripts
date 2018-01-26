# First part of the script prints INDAT and CHARGE
# parts of input file into individual text files
import os
import re
with open('FMO_Example.inp') as infile, open('output.txt', 'w') as outfile:
    copy = False
    a=re.compile(r"\bICHARG\b")
    b=re.compile(r"\bFRGNAM\b")
    for line in infile:
        if a.search(line) != None:
            copy = True
        if copy: 
            outfile.write(line)
        if b.search(line) != None:
            copy = False
with open('output.txt') as infile, open('charg.txt', 'w') as outfile:
     for line in infile :
         if 'FRGNAM' not in line:
            outfile.write(line)
os.remove('output.txt')

with open('FMO_Example.inp') as infile, open('indat_list_raw.txt', 'w') as outfile:
    copy = False
    a=re.compile(r"\bINDAT\b")
    b=re.compile(r"\bEND\b")
    for line in infile:
        if a.search(line) != None:
            copy = True
        if copy: 
            outfile.write(line)
        if b.search(line) != None:
            copy = False
with open('indat_list_raw.txt') as infile, open('indat_list.txt', 'w') as outfile:
     for line in infile :
         if 'INDAT' not in line and 'END' not in line:
            outfile.write(line)

# Second part of the script generates the arrays representing 
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

# Third part of the script numerates nonzero fragments.
# Prints indat for negative, positive and zero-charge fragments. 

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
    # loop over the elements and add spaces
    for i in line.split():
        # append to the list
        arr.append(i)
f4 = open('fragments_listing.txt', 'w')
f3 = list(enumerate(arr))
print >> f4, f3

arr2 = []
f5 = open('positive_charged_fragments_id.txt', 'w')
for i, j in f3:
    if j == '1':
     arr2.append(i)
     
f5 = open('positive_charged_fragments_id.txt', 'w')
for i, j in f3:
    if j == '1':
     print >> f5, i+1

arr3 = []
f6 = open('negative_charged_fragments_id.txt', 'w')
for i, j in f3:
    if j == '-1':
     arr3.append(i)          
     
f6 = open('negative_charged_fragments_id.txt', 'w')
for i, j in f3:
    if j == '-1':
     print >> f6, i+1
          
f4.close()
f5.close()
f6.close()

z1 = open('indat_list_positive.txt','w')
f7 = open('indat_list.txt')
lines=f7.readlines()
for i in arr2:
    print >> z1, lines[i]
    
z1.close()
f7.close()

z2 = open('indat_list_negative.txt','w')
f7 = open('indat_list.txt')
lines=f7.readlines()
for i in arr3:
    print >> z2, lines[i]
    
z2.close()
f7.close()

f4 = open('fragments_listing.txt', 'w')
f3 = list(enumerate(arr))
print >> f4, f3

arr4 = []
f8 = open('zero_charged_fragments_id.txt', 'w')
for i, j in f3:
    if j == '0':
     arr4.append(i)          
     
f9 = open('zero_charged_fragments_id.txt', 'w')
for i, j in f3:
    if j == '0':
     print >> f9, i+1
          
f4.close()
f8.close()
f9.close()

x2 = open('indat_list_zero.txt','w')
x7 = open('indat_list.txt')
lines=x7.readlines()
for i in arr4:
    print >> x2, lines[i]
    
x2.close()
f7.close()

os.remove('charg_list.txt')
os.remove('charg_formated.txt')
os.remove('fragments_listing.txt')
