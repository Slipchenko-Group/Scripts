import pandas as pd
import numpy as np
import math
import os
from other_functions import *


def get_amplitudes_df(geo, total_states, logs):

    """Construct a dataframe with the leading amplitudes for 
       each state (rows) for every scheme (columns)"""

    leading_amps = {}
    for log in logs:
        
        log_amps = {}
        for state in range(1, total_states + 1):
            log_amps[f'S{state}'] = []  
        state = 1  
        log_lines = open(f'{os.getcwd()}/{geo}/{log}', 'r').readlines()
        scheme_name = log.replace(f'_{geo}.log', '')
        
        while state < total_states + 1:    
            amps_n_orbs = []
            abs_amps = []
            
            for line in log_lines:
                if 'EXCITED STATE  ' in line and \
                f' {state}  ENERGY=   ' in line:
                    state_indx = log_lines.index(line)
                    amp_indx = state_indx+6
                    while '---' not in log_lines[amp_indx]:
                        amp = float(log_lines[amp_indx].split()[-1])
                        occ = int(log_lines[amp_indx].split()[0])
                        vir = int(log_lines[amp_indx].split()[1])
                        amps_n_orbs.append([amp, occ, vir])
                        abs_amps.append(abs(amp))
                        amp_indx += 1
            for amplitude in amps_n_orbs:
                if abs(amplitude[0]) == max(abs_amps):
                    log_amps[f'S{state}'].append(amplitude[1])
                    log_amps[f'S{state}'].append(amplitude[2])
            state += 1
        leading_amps[f'{scheme_name}'] = log_amps

    return(pd.DataFrame(leading_amps))


def get_states_dict(total_states, scheme, geo):

    """For the scheme, return a dictionary with the data
       describing each state: excitation energy (eV), 
       oscillator strenght and transition dipole"""

    log_lines = open(f'{os.getcwd()}/{geo}/{scheme}_{geo}.log', 'r').readlines()
    name = scheme
    
    scheme_dict = {}
    state = 1
    while state < total_states + 1:
        for line in log_lines:
            if f' TRANSITION FROM THE GROUND STATE TO EXCITED STATE' \
            in line and f' {state}\n' in line:
                i = log_lines.index(line)
                eline = log_lines[i+3]
                trdline = log_lines[i+7]
                osline = log_lines[i+8]
                energy_ev = 27.2114*(float(eline.split()[-1])-\
                                     float(eline.split()[-2]))
                transition_dipole = [float(trd) for trd in trdline.split()[3:6]]
                os_strength = float(osline.split()[-1])
                scheme_dict[f'S{state}'] = {}
                scheme_dict[f'S{state}']['transition_dipole'] = transition_dipole
                scheme_dict[f'S{state}']['os_strenght'] = os_strength
                scheme_dict[f'S{state}']['excitation_ev'] = energy_ev
        state += 1   
    return(scheme_dict)


def qm2efp_orbs(qm_dict, orbitals, scheme):

    """Match the QM orbitals for each state's leading amplitude to
       QM/EFP orbitals and add to dictionary with QM data"""

    for key in qm_dict:
        occ = qm_dict[key]['orbs'][0]
        vir = qm_dict[key]['orbs'][1] 

        scheme_occ = orbitals.loc[occ, scheme] if \
                     occ in orbitals.index else np.nan
        scheme_vir = orbitals.loc[vir, scheme] if \
                     vir in orbitals.index else np.nan
        qm_dict[key]['scheme_orbs'] = [scheme_occ, scheme_vir]

    missing_qmefp = []
    for qm_state in qm_dict:
        if np.nan in qm_dict[qm_state]['scheme_orbs']:
            missing_qmefp.append(str(f'{qm_state}'))

    for state in missing_qmefp:
        del qm_dict[state]
    
    return(qm_dict)



def match_states(geo, scheme, orbitals_csv, total_states, 
                 mag_thr, per_thr, analysis, transitions):

    """Find all the QM/EFP stats that match a single QM state.
       There are three checks for states to be matched: orbitals, 
       oscillator strenght and transition dipole moment.
       Then a single QM/EFP match is selected through either analysis 1 or 2"""

    # First get transitions, orbitals and dicts with data for each excitation
    # For both the QM file and the QM/EFP file.

    orbitals = pd.read_csv(f'{os.getcwd()}/{geo}/{orbitals_csv}', 
                           index_col='qm')
    
    qm_dict = get_states_dict(total_states, 'fullqm', geo)
    scheme_dict = get_states_dict(total_states, scheme, geo)
    
    for key in qm_dict:
        qm_dict[key]['orbs'] = transitions.loc[key, 'fullqm']

    for key in scheme_dict:
        scheme_dict[key]['orbs'] = transitions.loc[key, scheme]
        
    qm_dict = qm2efp_orbs(qm_dict, orbitals, scheme)
    
    for key in qm_dict:
        occ = qm_dict[key]['orbs'][0]
        vir = qm_dict[key]['orbs'][1]
        chr_occ = orbitals.loc[occ, 'chr']
        chr_vir = orbitals.loc[vir, 'chr']
        qm_dict[key]['transition_type'] = ''.join([chr_occ, chr_vir]) 

    # Now match all possible QM/EFP states to a QM state.
    perc1 = lambda a, b: abs(((float(a)-float(b))*100)/float(a))
    perc2 = lambda a, b: abs(((float(a)-(-1*float(b)))*100)/float(a))

    prematched = {}

    for state in qm_dict:

        qm_state = int(state[1:])
        scheme_state = 1
        matches = {}

        while scheme_state < total_states + 1:
            
            if f'S{scheme_state}' in scheme_dict:
                
                qm_info = qm_dict[f'S{qm_state}']
                scheme_info = scheme_dict[f'S{scheme_state}']
                qm_os = qm_info['os_strenght']
                scheme_os = scheme_info['os_strenght']
                td_index = qm_info['transition_dipole'].\
                            index(max(qm_info['transition_dipole'], key=abs))
                qm_td = qm_info['transition_dipole'][td_index]
                scheme_td = scheme_info['transition_dipole'][td_index]

                # first check: orbitals
                # continue with other checks only if this is true
                if qm_info['scheme_orbs'] == scheme_info['orbs']:

                    # second check: oscillator strenght
                    # matched if both strenghts < mag_thr
                    if abs(qm_os) < mag_thr and abs(scheme_os) < mag_thr:
                        matches[f'{scheme_state}'] = [0,0]
                        #matches.append(scheme_state)                    
                    # continue with third check if the error < per_thr
                    elif perc1(qm_os, scheme_os) < per_thr or \
                         perc2(qm_os, scheme_os) < per_thr:

                        # third check: transition dipole
                        # matched if both components < mag_thr
                        if abs(qm_td) < mag_thr and abs(scheme_td) < mag_thr:
                            matches[f'{scheme_state}'] = \
                                [min(perc1(qm_os, scheme_os), 
                                     perc2(qm_os, scheme_os)),0]
                            #matches.append(scheme_state)
                        # also matched if error < per_thr
                        elif perc1(qm_td, scheme_td) < per_thr or \
                             perc2(qm_td, scheme_td) < per_thr:
                            matches[f'{scheme_state}'] = \
                                [min(perc1(qm_os, scheme_os), 
                                     perc2(qm_os, scheme_os)),
                                 min(perc1(qm_td, scheme_td),
                                     perc2(qm_td, scheme_td))]
                            #matches.append(scheme_state)                        

            scheme_state += 1

        empty_dict = not matches
        if empty_dict == False:
            prematched[qm_state] = matches

    # Now match just one QM state with one QM/EFP state.
    # Two possible ways.
    if analysis == 1:
        matched = pd.DataFrame(analysis1(prematched, scheme))
    elif analysis == 2:
        A2_1 = analysis2(prematched)
        A2_2 = analysis2(A2_1)
        A2 = {'qm_num' : [], f'{scheme}_num' : []}
        for qm_state in A2_2:
            scheme_num = [int(scheme_state) for scheme_state in \
                          list(A2_2[qm_state].keys())]
            A2['qm_num'].append(qm_state)
            A2[f'{scheme}_num'].append(scheme_num[0])
        matched = (pd.DataFrame(A2))

    # Create final dictionary with the matched states and their data.
    matched_states_dict = {'qm' : [], 'qm_num' : [], 'transition_type' : [],
                f'{scheme}' : [], f'{scheme}_num' : []}

    for state in range(0, matched.shape[0]):
        qm_state = int(matched.iloc[state, 0])
        qm_info = qm_dict["".join(f'S{qm_state}')]
        scheme_state = int(matched.iloc[state, 1])
        scheme_info = scheme_dict["".join(f'S{scheme_state}')]
        append_excitations(scheme, matched_states_dict, qm_state, 
                           qm_info, scheme_state, scheme_info)

    return(pd.DataFrame(matched_states_dict))



def analysis1(prematched, scheme):

    """Match the QM/EFP states to EFP. This function
       stops looking at more QM/EFP states 
       once a match is found."""

    A1 = {'qm_num' : [], f'{scheme}_num' : []}
    for qm_state in prematched:
        scheme_states = list(int(state) for state in \
                            (prematched[qm_state].keys()))
        scheme_states.sort()
        state_index = 0
        while state_index < len(scheme_states):
            if scheme_states[state_index] not in A1[f'{scheme}_num']:
                A1[f'{scheme}_num'].append(scheme_states[state_index])
                A1['qm_num'].append(qm_state)
                break
            else:
                state_index += 1
    return(A1)


def analysis2(prematched):

    """Match the QM/EFP states to EFP. This function
       loops over all possible matches and finds the 
       match with smallest errors, prioritizing the 
       oscillator strenght and then the transition dipole"""
    
    matched = {}
    error_type = lambda x: 'type1' if x[0]==0 and x[1]==0 else\
                           ('type2' if x[0]!=0 and x[1]==0\
                            else'type3')

    for qm_state in prematched:    
        matched_states = prematched[qm_state] 
        type1 = [int(state) for state in matched_states \
                 if error_type(matched_states[state]) == 'type1']
        type2 = [state for state in matched_states \
                 if error_type(matched_states[state]) == 'type2']
        type3 = [state for state in matched_states \
                 if error_type(matched_states[state]) == 'type3']  

        if len(type1) > 0:
            best_match = min(type1)
            add2dict(matched, int(best_match), 
                    {f'{qm_state}' : prematched[qm_state][f'{best_match}']})

        elif len(type2) > 0:
            best_error = min([prematched[qm_state][state] for state in type2])
            best_match = [state for state in prematched[qm_state].keys() if \
                              prematched[qm_state][state] == best_error][0]
            add2dict(matched, int(best_match), 
                    {f'{qm_state}' : prematched[qm_state][best_match]})

        elif len(type3) > 0:
            best_error = min([prematched[qm_state][state] for state in type3])
            best_match = [state for state in prematched[qm_state].keys() if \
                              prematched[qm_state][state] == best_error][0]
            add2dict(matched, int(best_match), 
                    {f'{qm_state}' : prematched[qm_state][best_match]})

    return(matched)


def combine_data(geos, total_states, 
                 mag_thr, per_thr, analysis, schemes):

    """This functions combines the data for the 
       desired geometries and schemes"""

    dfs = []

    for geo in geos:

        qmefp_orbs = f'qmefp_{geo}_orbitals.csv'
        gas_orbs = f'gas_{geo}_orbitals.csv'

        logs = [f'{scheme}_{geo}.log' for scheme in schemes]
        logs.extend([f'gas_{geo}.log', f'fullqm_{geo}.log'])

        transitions = get_amplitudes_df(geo, total_states, logs)
        transitions['indexNumber'] = [int(idx.split('S')[-1]) for \
                                      idx in transitions.index]
        transitions.sort_values(['indexNumber'], ascending = [True], 
                                inplace = True)

        df = match_states(geo, 'gas', gas_orbs, total_states, 
                          mag_thr, per_thr, analysis, transitions)

        for scheme in schemes:
            scheme_df = match_states(geo, scheme, qmefp_orbs, total_states, 
                                     mag_thr, per_thr, analysis, transitions)
            df = df.merge(scheme_df, how='outer', 
                          on=['qm', 'qm_num', 'transition_type'])

        df['geo'] = f'{geo}'
        state_totalE = get_qm_state_energy(f'{geo}')
        geo_df = df.merge(state_totalE, how='left', 
                          left_on = ['qm_num'], right_on = ['state'])

        geo_df['num_nan'] = geo_df[schemes].isnull().sum(axis=1)

        geo_df = geo_df[geo_df['num_nan']<6]
        geo_df = geo_df.sort_values(by=['qm_num'])

        dfs.append(geo_df)

    result = pd.concat(dfs)
    return(result)


if __name__ == "__main__":

    geos = [geo for geo in os.listdir(os.getcwd()) if '.' not in geo\
            and len(geo)<4]
    	
    total_states = 15
    mag_thr = 0.1
    per_thr = 25
    schemes = ['full', 'semi', 'zero_dispva', 'zero_dispev',
                'zero', 'semiz',
                'zero_exrep', 'zero_screen',]
    
    A1 = combine_data(geos, total_states, mag_thr, per_thr, 1, schemes)
    A2 = combine_data(geos, total_states, mag_thr, per_thr, 2, schemes)
    result = pd.merge(A1, A2, how='inner')
    result.to_csv('gamess_results.csv', index=False)
