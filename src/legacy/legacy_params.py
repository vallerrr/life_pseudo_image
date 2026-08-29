"""Preserved pre-reorganisation prototype. Not part of the current pipeline.

Superseded by the simulation-first first deliverable (see doc/decisions_shared.md
D01, D06). Kept because it encodes the HRS variable selection and recode maps, which
are expensive to reconstruct. Paths come from the registry so it still resolves.
"""

import re

from utils import shared_paths as P

current_path = P.PROJECT_ROOT
# hrs example
hrs_path = P.HRS_DIR / "randhrs1992_2022v1_STATA" / "randhrs1992_2022v1.dta"
hrs_var_selection = {'upper':['r1work','R1LBRF','RABYEAR','R1AGEM_B','RAGENDER','RARACEM','RAEDYRS','RAEDUC'],
                     'middle':['R1MPART','R1MSTATH','H1ITOT','H1HHRES'],
                     'lower':['R1BMI','R1DRINK','R3DRINKD','R1SMOKEV','R1SMOKEN']}

replace_dict = {'r1work': {'1.working for pay': -1, '0.not working for pay': 1},
                'r1lbrf': {'1.works ft': 1, '7.not in lbrf': 7, '6.disabled': 6, '3.unemployed': 3, '5.retired': 5, '4.partly retired': 4, '2.works pt': 2},
                'ragender': {'2.female': -1, '1.male': 1},
                'raracem': {'1.white/caucasian': 1, '3.other': 3, '2.black/african american': 2},
                'raedyrs': {'16.0': 2, '12.0': 6, '13.0': 5, '17.17+ yrs': 1, '8.0': 10, '10.0': 8, '15.0': 3, '7.0': 11, '11.0': 7, '14.0': 4, '9.0': 9, '5.0': 13, '6.0': 12, '3.0': 15, '4.0': 14, '2.0': 16, '0.none': 18, '1.0': 17},
                'raeduc': {'5.college and above': 1, '3.high-school graduate': 3, '4.some college': 2, '1.lt high-school': 5, '2.ged': 4},
                'r1mpart': {'0.no': -1, '1.yes': 1},
                'r1mstath': {'1.married': 1, '5.divorced': 5, '8.never married': 8, '7.widowed': 7, '6.separated/divorced':6,'4.separated': 4, '2.married,spouse absent': 2, '9.unknown unmar': 8}, 'r1drink': {'1.yes': 1, '0.no': -1}, 'r3drinkd': {'0.0 or doesnt drink': 0, '3.0': 3, '4.0': 4, '7.0': 7, '2.0': 2, '1.0': 1, '6.0': 6, '5.0': 5}, 'r1smokev': {'0.no': -1, '1.yes': 1}, 'r1smoken': {'0.no': -1, '1.yes': 1}}


def convert_dict(target_dict):
    new_dict = {}
    for key in target_dict.keys():
        new_dict[key]=[x.lower() for x in target_dict[key]]
    return new_dict
def select_data(df,var_selection):
    columns = df.columns
    selected_cols = []
    for key in var_selection.keys():
        vars = [x.lower() for x in var_selection[key]]
        for var in vars:
            if len(re.findall(r'\d', var)) > 0: #multiple waves
                var_split = var.split(re.findall(r'\d', var)[0])
                pattern = re.compile(rf'{var_split[0]}\d+{var_split[1]}$')
                filtered_vars = [var for var in columns if pattern.match(var)]
            else:
                filtered_vars = [var]
            selected_cols+= filtered_vars
    return df[selected_cols]
