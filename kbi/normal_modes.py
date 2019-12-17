
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-prg', action='store', dest='prg', type=str, help='qchem or gamess')
parser.add_argument('-log', action='store', dest='log', type=str, help='output file')
parser.add_argument('-name', action='store', dest='name', type=str, help='name of csv output')
parser.add_argument('-linear', action='store', dest='linear', type=int, help='1 if linear, 0 otherwise')
arguments = parser.parse_args()
prg, log, out_name, linear = arguments.prg, arguments.log, arguments.name, arguments.linear

def QChem_modes(log, out_name, linear):

    # log = name of frequency file
    # out_name = desired name for output csv
    # linear = 0 if not linear and 1 if linear
    
    import pandas as pd
    import numpy as np

    inp = open(log, 'r')
    lns = inp.readlines()

    N = lns[lns.index('$molecule\n'):].index('$end\n') - 2
    
    if linear == 0:
        N_modes = 3*N-6
    else:
        N_modes = 3*N-5
        
    freqs = []
    red_mass = []
    atom_mass = []
    mdict = {}

    for ln in lns:
        if 'Mode' in ln:
            i = lns.index(ln)
            freqs.extend([float(f) for f in lns[(i+1)].split()[1:]])
            red_mass.extend([float(f) for f in lns[(i+3)].split()[2:]])
            modes = [int(m) for m in ln.split()[1:]]

            for mode in modes:
                mdict[mode] = []
                for ln in lns[i+8:i+8+N]:
                    mi = modes.index(mode)+1
                    mdict[mode].extend(float(c) for c in ln.split()[3*mi-2:3*mi+1])

        elif 'Has Mass' in ln:
            atom_mass.extend([(float(ln.split()[-1]))]*3)

    modes = pd.DataFrame(data=mdict)  

    # step 1 
    # multiply rows by the sqrt(atomic_mass) and columns by sqrt(reduced_mass) of the vibration)
    modes_mw = modes.mul([np.sqrt(am) for am in atom_mass],axis=0)\
                    .mul([1/np.sqrt(rm) for rm in red_mass],axis=1)


    # step 2

    # 3N-6 rows x 3N cols
    # divide columns by sqrt(atomic_mass) and rows by (0.172*sqrt(freq))
    modes_mwT_f = modes_mw.T.mul([1/np.sqrt(am) for am in atom_mass],axis=1)\
                            .mul([1/(0.172*np.sqrt(freq)) for freq in freqs],axis=0)

    # 3N rows x 3N-6 cols
    # multiply rows by sqrt(atomic_mass) and cols by (0.172*sqrt(freq))
    modes_mw_f = modes_mw.mul([np.sqrt(am) for am in atom_mass], axis=0)\
                          .mul([(0.172*np.sqrt(freq)) for freq in freqs], axis=1)

    unity = round(modes_mwT_f.dot(modes_mw_f),1)

    if list(unity.sum(axis=0)) == [1]*N_modes:
        modes_mwT_f.to_csv(f'{out_name}',header=False,index=False)
        modes_mw_f.to_csv(f'{out_name}_2', header=False, index=False)
    else:
        print('Error: D*D is not identity')

if prg=='qchem':
    QChem_modes(log, out_name, linear)
else:
    print('missing gamess function')
