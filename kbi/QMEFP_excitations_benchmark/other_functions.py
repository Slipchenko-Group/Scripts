import pandas as pd
import numpy as np
import math
import os

def append_excitations(scheme, out_dict, qm_state, qm_info, scheme_state, scheme_info):
    
    out_dict['qm'].append(round(qm_info['excitation_ev'], 8))
    out_dict['qm_num'].append(qm_state)
    out_dict[f'{scheme}'].append(round(scheme_info['excitation_ev'], 8))
    out_dict[f'{scheme}_num'].append(scheme_state)
    out_dict['transition_type'].append(qm_info['transition_type'])
    return(out_dict)


def get_qm_state_energy(geo):
 
    """Get the total energy for each state. 
       This function can/should be combined with get_amplitudes_df."""

    states_energy = {'state' : [], 'state_energy' : []}
    qm_lns = open(f'{os.getcwd()}/{geo}/fullqm_{geo}.log', 'r').readlines()   
    for line in qm_lns:
        if 'EXCITED STATE' in line and 'ENERGY=' in line:
            state = float(line.split()[2])
            energy = float(line.split()[4])
            states_energy['state'].append(state)
            states_energy['state_energy'].append(round(energy, 5))
    
    return(pd.DataFrame(data=states_energy))


def add2dict(in_dict, key, val):
    if key not in in_dict.keys():
        in_dict[key] = val
    else:
        in_dict[key].update(val)
    return(in_dict)

