import pandas as pd
import os


def get_n_solv_atoms(geo):

    """Read GAMESS files (fullqm and gas) to figure out the number of
       atoms in the chromophore and the solvent."""

    gas = open(f'{os.getcwd()}/{geo}/gas_{geo}.log').readlines()
    fullqm = open(f'{os.getcwd()}/{geo}/fullqm_{geo}.log').readlines()
    logs = [fullqm, gas]
    n_atoms = []
    for log in logs:
        for line in log:
            if '$data' in line:
                start_atom = log.index(line)+3
            elif 'WORDS OF MEMORY AVAILABLE' in line:
                end_atom = log.index(line)-1
        n_atoms.append(end_atom - start_atom)
    n_solv_atoms = n_atoms[0] - n_atoms[1]
    n_atoms.append(n_solv_atoms)
    return(n_atoms)

def get_nto_df(geo):

    """Read QChem output files with NTO analysis and returns dataframe
       indicating which states have charge transfer to/from solvent:
       true if there is 0.01 or more charge located on the 
       electron or electron hole according to the Mulliken pop. 
       analysis."""
    
    atoms = get_n_solv_atoms(geo)
    na_solute, na_solvent = atoms[1], atoms[2]
    out_lines = open(f'{os.getcwd()}/cis_nto/{geo}_cis.out', 'r').readlines()
    data = {'state' : [], 'h+_chtr' : [], 'e-_chtr' : [], 
            'state_energy' : [], 'multiplicity' : [],}
    state = 1

    for line in range(0, len(out_lines)):        
        if 'Excited state' and 'excitation energy ' in out_lines[line]:
            energy_line = out_lines[line+1].split()
            hf_energy = float(energy_line[-2])
            mult = out_lines[line+2].split()
            if mult[-1] == 'Singlet':
                multiplicity = 1
            else:
                multiplicity = 0              
            data['state_energy'].append(hf_energy)
            data['multiplicity'].append(multiplicity)
   
        if 'Mulliken Population Analysis' in out_lines[line]:
            data['state'].append(float(state))         
            solv_init_idx = int(line + 3 + na_solute)
            solv_end_idx = int(solv_init_idx + na_solvent)
            hole = []
            electron = []
            for line_idx in range(solv_init_idx, solv_end_idx):
                chtr_h = float(out_lines[line_idx].split()[3])
                chtr_e = float(out_lines[line_idx].split()[4])
                hole.append(chtr_h)
                electron.append(chtr_e)
                
            hole_chtr = (1 if any(abs(x) > 0.01 for x in hole) else 0)
            elec_chtr = (1 if any(abs(x) > 0.01 for x in electron) else 0)
            data['h+_chtr'].append(hole_chtr)
            data['e-_chtr'].append(elec_chtr)
            state += 1
    
    data['geo'] = [f'{geo}'] * len(data['state'])
    return (pd.DataFrame(data=data))


if __name__ == "__main__":

    geos = [geo for geo in os.listdir(os.getcwd()) if '.' not in geo\
            and len(geo)<4]

    dfs = []
    for geo in geos:
        dfs.append(get_nto_df(geo))
    result = pd.concat(dfs)
    result.to_csv('NTO_analysis.csv', index=False)
