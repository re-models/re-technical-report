
from ast import literal_eval
from os import getcwd, path
from pathlib import Path
import tarfile

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Set, List
import random
from typing import Set, List


nice_model_names = {'StandardGlobalReflectiveEquilibrium':'QuadraticGlobalRE',
                     'StandardLocalReflectiveEquilibrium':'QuadraticLocalRE',
                     'StandardGlobalReflectiveEquilibriumLinearG': 'LinearGlobalRE',
                     'StandardLocalReflectiveEquilibriumLinearG': 'LinearLocalRE'
                    }
nice_model_short_names = {'StandardGlobalReflectiveEquilibrium':'QGRE',
                     'StandardLocalReflectiveEquilibrium':'QLRE',
                     'StandardGlobalReflectiveEquilibriumLinearG': 'LGRE',
                     'StandardLocalReflectiveEquilibriumLinearG': 'LLRE'
                    }

def literal_eval_cols(data: DataFrame, cols: List[str]):
    for col_name in cols:
        data[col_name] = data.apply(lambda x: literal_eval(x[col_name]), axis=1)

# replace string 'set()' for literal_eval to work  properly
def replace_set(row, label):
    return row[label].replace('set()', '{}')

def load_re_data(data_dir, 
                 data_file_name,
                 usecols=None,
                 evalcols=None,
                 filter_erroneous_runs:bool = False,
                 replace_model_names = True):
    if data_file_name[data_file_name.find('.'):len(data_file_name)] == '.csv.tar.gz':
        with tarfile.open(path.join(data_dir,data_file_name)) as tar:
            for tarinfo in tar:
                file_name = tarinfo.name
            tar.extractall(data_dir)        
     
        re_data = pd.read_csv(path.join(data_dir, file_name), usecols=usecols)
    else:
        re_data = pd.read_csv(path.join(data_dir, data_file_name), usecols=usecols)

    if usecols is None or 'global_optima' in usecols:
        re_data['global_optima'] = re_data['global_optima'].replace('set()', '{}')
        
    if usecols is None or 'fixed_points' in usecols:
        re_data['fixed_points'] = re_data['fixed_points'].replace('set()', '{}')
         
    if evalcols is None:
        evalcols = ['global_optima', 'go_coms_consistent', 'go_union_consistent',
                    'go_full_re_state', 'go_fixed_point', 'go_account', 'go_faithfulness',
                    'fixed_points', 'fp_coms_consistent', 'fp_union_consistent', 'fp_full_re_state',
                    'fp_account', 'fp_faithfulness','fp_global_optimum', 'coms_evolution', 'init_coms',
                    'go_union_consistent', 'fp_union_consistent']
    # filter for cols that are being used
    if usecols is not None:
        evalcols = [use_col for use_col in usecols if use_col in evalcols]

    
    # WARNING: XXX (eval not being save)
    # converting mere strings to data-objects
    literal_eval_cols(re_data, evalcols)
    
    # if columns 'initcoms' was initialised as set we make it a frozen set, which is need for 
    # some of the analysis routines 
    if (usecols is None or 'init_coms' in usecols) and ('init_coms' in evalcols):
        re_data['init_coms'] = re_data.apply(lambda x: frozenset(x['init_coms']), axis=1)
        

    if replace_model_names:
        # Adding model short names
        re_data['model_short_name'] = re_data['model_name'].map(lambda x: nice_model_short_names[x])
        re_data['model_name'] = re_data['model_name'].map(lambda x: nice_model_names[x])


    if filter_erroneous_runs:
        return re_data.loc[re_data['error_code'].isna()]
    else:
        return re_data

def heatmap_plot(*args, **kwargs):
    data = kwargs.pop('data')
    mask = kwargs.pop('mask')
    annot_std = kwargs.pop('annot_std')
    annot_std_fmt = kwargs.pop('annot_std_fmt')
    annot_fmt = kwargs.pop('annot_fmt')
    values = kwargs.pop('values')
    index = kwargs.pop('index')
    columns = kwargs.pop('columns')
    # labels = x_mean
    cmap = plt.get_cmap('coolwarm')
    cmap.set_bad('white')
    x_mean = pd.pivot_table(data, index=[index], columns=columns,
                            values=values, aggfunc=np.mean)
    if annot_std:
        x_std = pd.pivot_table(data, index=[index], columns=columns,
                               values=values, aggfunc=np.std)
        labels = x_mean.applymap(lambda x: annot_fmt.format(x)) + x_std.applymap(lambda x: annot_std_fmt.format(x))
        sns.heatmap(x_mean, cmap=cmap, mask=mask, annot=labels, fmt='', **kwargs)
    else:
        sns.heatmap(x_mean, cmap=cmap, mask=mask, annot=x_mean, **kwargs)


def normalized_heat_maps_by_weights(re_data, values, title, index='weight_account', columns='weight_systematicity',
                         annot_std=False, annot_fmt="{:2.0f}\n", annot_std_fmt=r'$\pm${:2.1f}', vmin=0, vmax=1,
                         output_dir=None, file_name=None, index_label=r'$\alpha_A$', columns_label=r'$\alpha_S$'):
    g = sns.FacetGrid(re_data, col='model_name', col_wrap=2, height=5, aspect=1)
    g.fig.suptitle(title, y=1.01)
    mask = pd.pivot_table(re_data, index=[index], columns=columns,
                          values=values, aggfunc=np.mean).isnull()
    g.map_dataframe(heatmap_plot, cbar=False, mask=mask, values=values, index=index, columns=columns,
                    annot_std=annot_std, annot_fmt=annot_fmt, annot_std_fmt=annot_std_fmt, vmin=vmin, vmax=vmax)
    g.set_axis_labels(columns_label, index_label)
    g.set_titles("{col_name}")
    if (file_name is not None) and (output_dir is not None):
        g.savefig(path.join(output_dir, file_name + '.pdf'), bbox_inches='tight')
        g.savefig(path.join(output_dir, file_name + '.png'), bbox_inches='tight')

def plot_multiple_error_bars(data, var_y, ylabel,  
                             var_hue = 'model_name', hue_title='Model',
                             var_std = None,
                             var_x = 'n_sentence_pool', xlabel = 'n', xticks=[6,7,8,9],
                             file_name=None, output_dir=None,
                             jitter=True, jitter_size=0.03,
                             bbox_to_anchor=(1., 0.2)):
    # If no col for error bars is given, we assume that the data is not aggregated and use `describe()` to do so
    if var_std is None:
        groupby = [var_hue] + [var_x]
        data_summary = data.groupby(groupby)[var_y].describe().reset_index()
        var_y = 'mean'
        var_std ='std'
    else:
        data_summary = data
    
    if jitter:
        data_summary[var_x] = data_summary.apply(lambda x: x[var_x]+random.uniform(-jitter_size, jitter_size), axis=1)

    for name, group in data_summary.groupby(var_hue):
        #display(group)
        plt.errorbar(group[var_x], 
                     group[var_y],
                     yerr=group[var_std],
                     marker='o', 
                     #linestyle='', # providing an empty string omits lines 
                     markersize=5,
                     capsize=3,
                     label=name)

    plt.legend(loc='center left', bbox_to_anchor=bbox_to_anchor, ncol=1, title=hue_title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(xticks)

    if (file_name is not None) and (output_dir is not None):
        plt.savefig(path.join(output_dir, file_name + '.pdf'), bbox_inches='tight')
        plt.savefig(path.join(output_dir, file_name + '.png'), bbox_inches='tight')

def random_weights():
    
    # full range
    alpha = random.uniform(0, 1)
    beta = random.uniform(0,1)
    
    # conversion to weights
    acc = (alpha * beta)/(alpha + beta - alpha * beta)
    sys = (beta - alpha * beta)/(alpha + beta - alpha * beta)
    
    fai = 1 - (acc + sys)
    
    return (acc, sys, fai)

def apply_fun_to_adjacents(li: List[any],fun) -> List[any]:
    if len(li) == 1:
        return []
    else: 
        return [fun(li[0], li[1])] + apply_fun_to_adjacents(li[1:], fun)
    
def simple_hamming(x:Set, y:Set) -> float:
    return len(x.union(y).difference(x.intersection(y)))

def mean_simple_hamming_adjacents(li:List[Set[int]]) -> float:
    ret = apply_fun_to_adjacents(li, simple_hamming)
    return 0 if len(ret) == 0 else sum(ret)/len(ret) 

def mean_d_init_coms_go(row):
    init_coms = row['init_coms']
    d_init_coms_go = [simple_hamming(init_coms, go_coms) 
                      for go_theory, go_coms in row['global_optima']]
    return sum(d_init_coms_go)/len(d_init_coms_go)