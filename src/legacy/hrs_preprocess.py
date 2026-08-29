import pandas as pd
from legacy import legacy_params as params
import re
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore')
current_path = params.current_path
var_selection = params.convert_dict(params.hrs_var_selection)

df_selected = params.select_data(pd.read_stata(params.hrs_path), var_selection).dropna()



for var in params.replace_dict.keys():
    if len(re.findall(r'\d', var)) > 0:  # multiple waves
        var_split = var.split(re.findall(r'\d', var)[0])
        pattern = re.compile(rf'{var_split[0]}\d+{var_split[1]}$')
        filtered_vars = [var for var in df_selected.columns if pattern.match(var)]
        df_selected.loc[:, filtered_vars] = df_selected.loc[:, filtered_vars].astype(str).replace(params.replace_dict[var])
    else:
        df_selected.loc[:, var] = df_selected.loc[:, var].astype(str).replace(params.replace_dict[var])

df_selected.to_csv(current_path / 'file/sample_data/hrs_preprocessed.csv', index=False)

df_selected = pd.read_csv(current_path / 'file/sample_data/hrs_preprocessed.csv')



def normalize_across_time(df, columns):
    """
    normalisation across different time
    :param df:
    :param columns:
    :return:
    """
    for col in columns:
        # Generate the column names for the specified time periods
        if len(re.findall(r'\d', col)) > 0:  # multiple waves
            var_split = col.split(re.findall(r'\d', col)[0])
            pattern = re.compile(rf'{var_split[0]}\d+{var_split[1]}$')
            filtered_vars = [var for var in df_selected.columns if pattern.match(var)]
            # Calculate the global minimum and maximum across these columns
            global_min = df[filtered_vars].min().min()
            global_max = df[filtered_vars].max().max()

            # Normalize the specified columns
            df[filtered_vars] = df[filtered_vars].apply(lambda x: (x - global_min) / (global_max - global_min), axis=0)

        else:
            min_val,max_val = min(df[col]),max(df[col])
            df[col] = (df[col]-min_val)/(max_val-min_val)

    return df

columns = []
for key in var_selection.keys():
    columns += var_selection[key]

df_selected = normalize_across_time(df_selected, columns)

# normalise data
# reformat the data by time

period = 14
row_index = 0

for row_index in range(0,21):
    row_reformatted = pd.DataFrame(columns=['var_name'] + [f't{i}' for i in range(1, period + 1)])
    one_row = df_selected.loc[row_index,:]
    for col in columns:
        if len(re.findall(r'\d', col)) > 0:  # multiple waves
            var_split = col.split(re.findall(r'\d', col)[0])
            pattern = re.compile(rf'{var_split[0]}\d+{var_split[1]}$')
            filtered_vars = [var for var in df_selected.columns if pattern.match(var)]
            if len(filtered_vars) <= period:
                periods = [int(re.findall(r'\d+', x)[0]) for x in filtered_vars]
                values = [None if x not in periods else one_row[f'{var_split[0]}{x}{var_split[1]}'] for x in range(1,period+1)]
            else:
                values = list(one_row[filtered_vars])
            row_reformatted.loc[len(row_reformatted),] = [col]+values
        else:
            row_reformatted.loc[len(row_reformatted),] = [col]+[one_row[col]]*period


    row_reformatted.set_index('var_name',inplace=True)
    row_reformatted=row_reformatted.fillna(0)


    '''
    # one color 
    plt.imshow(row_reformatted, cmap='gray', aspect='auto')
    plt.colorbar()  # Adds a colorbar to indicate the mapping from grey levels to data values
    plt.show()
    '''
    row_groups = {'upper':'Blues','middle':'Greens','lower':'Reds'}

    df= row_reformatted.copy()
    # Create a figure for the plots
    fig = plt.figure(figsize=(7, 11))
    positions = [[0.15, 0.67, 0.8, 0.3],  # Top plot
                 [0.15, 0.452, 0.8, 0.3],  # Middle plot
                 [0.15, 0.288, 0.8, 0.3]]
    for (key,cmap),position in zip(row_groups.items(),positions):
        ax = fig.add_axes(position)

        # Select the rows for the current group
        group_df = df.loc[var_selection[key]]
        # Plot the selected rows with the specified colormap
        cax = ax.imshow(group_df, cmap=cmap)
        ax.set_yticklabels(['']+var_selection[key])
        ax.axes.get_xaxis().set_visible(False)

        if key == 'upper':
            ax.spines['bottom'].set_visible(False)
        elif key == 'lower':
            ax.spines['top'].set_visible(False)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    plt.savefig(params.current_path/f'file/sample_graph/{row_index}.png')



'''
replace_dict = {}
for key in var_selection:
    vars = var_selection[key]
    for var in vars:
        if var not in replace_dict.keys():
            unique_values =list(df_selected[var].unique())
            replace_contrl = False if input(f'in var {var}, unique values are {unique_values}, replace? 0-> no')== '0' else True
            if replace_contrl:
                replace_dict[var]={}
                for unique_value in unique_values:
                    replace_val = int(input(f'for {unique_value},replace val='))
                    replace_dict[var][unique_value]=replace_val
'''
