
from ast import literal_eval
from math import log
from os import getcwd, path
from pathlib import Path
import tarfile

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
from typing import Set, List
import random
import scipy.stats as spst
from itertools import combinations

from IPython import get_ipython

from matplotlib_venn import venn2
from tau import DialecticalStructure, Position, StandardPosition


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

data_set_name_to_file_name = {'FULL': 're_data_tau_alpha',
                              'MINI': 're_data_tau_alpha_mini',
                              'TINY': 're_data_tau_alpha_tiny'}

coolwarm = ['#6788ee', 
            '#9abbff', 
            '#e26952', 
            '#f7a889']


def literal_eval_cols(data: DataFrame, cols: List[str]):
    for col_name in cols:
        data[col_name] = data.apply(lambda x: literal_eval(x[col_name]), axis=1)

# replace string 'set()' for literal_eval to work  properly
def replace_set(row, label):
    return row[label].replace('set()', '{}')

def re_data_by_name(data_name, 
                    usecols=None,
                    evalcols=None,
                    filter_erroneous_runs:bool = False,
                    replace_model_names = True,
                    data_dir=path.join(Path(getcwd()).parent.absolute(), "data"),
                    data_set_tiny_url="https://raw.githubusercontent.com/re-models/re-technical-report/main/data/re_data_tau_alpha_tiny.csv"):
    
    # running on colab
    if 'google.colab' in str(get_ipython()):
        # On Colab we can only load the 'TINY' data set.
        if data_name == 'TINY':
            return load_re_data(url=data_set_tiny_url,
                                usecols=usecols,
                                evalcols=evalcols,
                                filter_erroneous_runs=filter_erroneous_runs,
                                replace_model_names=replace_model_names)
        else:
           raise RuntimeWarning("It seems you are calling this method from Colab, which is confined to using the 'TINY' data set.")
    # not running on colab (using local file system to load data)
    else:
        # Checking whether we have an already unpacked data file (to reduce costly re-unpacking of archived data files)
        if path.exists(path.join(data_dir,f"{data_set_name_to_file_name[data_name]}.csv")):
            data_file_name = f"{data_set_name_to_file_name[data_name]}.csv"
        elif path.exists(path.join(data_dir,f"{data_set_name_to_file_name[data_name]}.tar.gz")):
            data_file_name = f"{data_set_name_to_file_name[data_name]}.tar.gz"
        else:
            msg = f"Requested data file does not exist. Make sure that the data file is in the corresponding" \
                  f"directory ({data_dir}) or use the function 'load_re_data' instead."
            raise RuntimeWarning(msg)

        return load_re_data(data_dir=data_dir,
                            data_file_name=data_file_name,
                            usecols=usecols,
                            evalcols=evalcols,
                            filter_erroneous_runs=filter_erroneous_runs,
                            replace_model_names=replace_model_names)


def load_re_data(data_dir=None,
                 data_file_name=None,
                 url=None,
                 usecols=None,
                 evalcols=None,
                 filter_erroneous_runs:bool = False,
                 replace_model_names = True):
    if url:
        re_data = pd.read_csv(url, usecols=usecols)
    elif data_file_name[data_file_name.find('.'):len(data_file_name)] == '.csv.tar.gz':
        with tarfile.open(path.join(data_dir,data_file_name)) as tar:
            for tarinfo in tar:
                file_name = tarinfo.name
            tar.extractall(data_dir)        
     
        re_data = pd.read_csv(path.join(data_dir, file_name), usecols=usecols)
    else:
        re_data = pd.read_csv(path.join(data_dir, data_file_name), usecols=usecols)

    # workaround to enable evaluation of string that correspond to empty sets
    if usecols is None or 'global_optima' in usecols:
        re_data['global_optima'] = re_data['global_optima'].replace('set()', '{}')
        
    if usecols is None or 'fixed_points' in usecols:
        re_data['fixed_points'] = re_data['fixed_points'].replace('set()', '{}')
         
    if evalcols is None:
        evalcols = ['global_optima', 'go_coms_consistent', 'go_union_consistent',
                    'go_full_re_state', 'go_fixed_point', 'go_account', 'go_faithfulness',
                    'fixed_points', 'fp_coms_consistent', 'fp_union_consistent', 'fp_full_re_state',
                    'fp_account', 'fp_faithfulness','fp_global_optimum', 
                    'coms_evolution', 'theory_evolution', 'init_coms',
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

#def heatmap_plot(*args, **kwargs):
#    data = kwargs.pop('data')
#    mask = kwargs.pop('mask')
#    annot_std = kwargs.pop('annot_std')
#    annot_std_fmt = kwargs.pop('annot_std_fmt')
#    annot_fmt = kwargs.pop('annot_fmt')
#    values = kwargs.pop('values')
#    index = kwargs.pop('index')
#    columns = kwargs.pop('columns')
#    # labels = x_mean
#    cmap = plt.get_cmap('coolwarm')
#    cmap.set_bad('white')
#    x_mean = pd.pivot_table(data, index=[index], columns=columns,
#                            values=values, aggfunc=np.mean)
#    if annot_std:
#        x_std = pd.pivot_table(data, index=[index], columns=columns,
#                               values=values, aggfunc=np.std)
#        labels = x_mean.applymap(lambda x: annot_fmt.format(x)) + x_std.applymap(lambda x: annot_std_fmt.format(x))
#        sns.heatmap(x_mean, cmap=cmap, mask=mask, annot=labels, fmt='', **kwargs)
#    else:
#        sns.heatmap(x_mean, cmap=cmap, mask=mask, annot=x_mean, **kwargs)

def heatmap_plot(*args, **kwargs):
    data = kwargs.pop('data')
    mask = kwargs.pop('mask')
    annot_std = kwargs.pop('annot_std')
    annot_std_fmt = kwargs.pop('annot_std_fmt')
    annot_fmt = kwargs.pop('annot_fmt')
    values = kwargs.pop('values')
    index = kwargs.pop('index')
    columns = kwargs.pop('columns')
    bootstrap = kwargs.pop('bootstrap')
    n_resamples = kwargs.pop('n_resamples')
    
    #set_heatmap_plot_style()
    
    if bootstrap:
        agg_mean = lambda x: bootstrap_mean(x, n_resamples=n_resamples)
        agg_std = lambda x: bootstrap_std(x, n_resamples=n_resamples)
        
    else:
        agg_mean = np.mean
        agg_std = np.std
    
    cmap = plt.get_cmap('coolwarm')
    cmap.set_bad('white')
    
    x_mean = pd.pivot_table(data, index=[index], columns=columns,
                            values=values, aggfunc=agg_mean)
    
    if annot_std:
        x_std = pd.pivot_table(data, index=[index], columns=columns,
                               values=values, aggfunc=agg_std)
        labels = x_mean.applymap(lambda x: annot_fmt.format(x)) + x_std.applymap(lambda x: annot_std_fmt.format(x))
        sns.heatmap(x_mean, cmap=cmap, mask=mask, annot=labels, fmt='', **kwargs)
    else:
        sns.heatmap(x_mean, cmap=cmap, mask=mask, annot=x_mean,
                    fmt=annot_fmt[annot_fmt.find('{')+2:annot_fmt.find('}')],
                    **kwargs)
        
    


def heat_maps_by_weights(re_data, values, title=None, index='weight_account', 
                         columns='weight_systematicity',
                         annot_std=False, annot_fmt="{:2.0f}\n", annot_std_fmt=r'$\pm${:2.1f}', vmin=0, vmax=1,
                         output_dir=None, file_name=None, index_label=r'$\alpha_A$', columns_label=r'$\alpha_S$', 
                         bootstrap=False, n_resamples=1000, col_model='model_name', col_order=None):
    
    set_heatmap_plot_style()
    
    g = sns.FacetGrid(re_data, col=col_model, col_wrap=2, height=5, aspect=1, col_order=col_order)
    if title:
        g.fig.suptitle(title, y=1.01)
    mask = pd.pivot_table(re_data, index=[index], columns=columns,
                          values=values, aggfunc=np.mean).isnull()
    g.map_dataframe(heatmap_plot, cbar=False, mask=mask, values=values, index=index, columns=columns,
                    annot_std=annot_std, annot_fmt=annot_fmt, annot_std_fmt=annot_std_fmt, vmin=vmin, vmax=vmax, 
                    bootstrap=bootstrap, n_resamples=n_resamples)
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
                             bbox_to_anchor=(1., 0.2),
                             alt_labels=None):
    
    n_hues = data[var_hue].nunique()
    set_errorbar_plot_style(n_hues, models=var_hue=="model_name")
    
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
        
    #sns.set(font_scale=1.25)

    for name, group in data_summary.groupby(var_hue):
        #display(group)
        if alt_labels:
            name=alt_labels[name]
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

def mean_simple_hamming_distance(li:List[Set[int]]) -> float:
    distances = [simple_hamming(el1, el2) for el1,el2 in combinations(li, 2)]
    # cases with one element only
    if len(distances)==0:
        return None
    return sum(distances)/len(distances)


def plot_venn(result_df, col_setA, col_setB, col_cut,
              label_setA, label_setB, 
              output_dir=None, file_name=None, rel_label=True):

    for i, row in result_df.iterrows():

        plt.subplot(2,2,i+1)

        subsets = (row[col_setA]-row[col_cut], row[col_setB]-row[col_cut], row[col_cut])

        v = venn2(subsets = subsets, set_labels=(label_setA + '\n' + str(row[col_setA]), 
                    label_setB+ '\n' + str(row[col_setB])))

        v.get_patch_by_id('10').set_color("b")
        v.get_patch_by_id('11').set_color("grey")
        if rel_label:
            rel_setA = row[col_cut]/row[col_setA]*100
            rel_setB = row[col_cut]/row[col_setB]*100
            v.get_label_by_id('11').set_text(f"{row[col_cut]} \n ({rel_setA:.0f}% / {rel_setB:.0f}%)")

        plt.title(row["model_name"], fontsize=14)
    
    if (file_name is not None) and (output_dir is not None):
        plt.savefig(path.join(output_dir, file_name + '.pdf'), bbox_inches='tight')
        plt.savefig(path.join(output_dir, file_name + '.png'), bbox_inches='tight')


def diff_heatmap_plot(*args, **kwargs):
    data = kwargs.pop('data')
    #mask = kwargs.pop('mask')
    annot_std = kwargs.pop('annot_std')
    annot_std_fmt = kwargs.pop('annot_std_fmt')
    annot_fmt = kwargs.pop('annot_fmt')
    values = kwargs.pop('values')
    index = kwargs.pop('index')
    columns = kwargs.pop('columns')
    bootstrap = kwargs.pop('bootstrap')
    # col that is used for the groupby for the row in the facetgrid
    # (will be used to generate a title)
    facet_row = kwargs.pop('facet_row')
    row_title = kwargs.pop('row_title')
    if facet_row:
        # should be unique
        row_entry = data[facet_row].unique()[0]
    else:
        row_entry = None
    # labels = x_mean
    cmap = plt.get_cmap('coolwarm')
    cmap.set_bad('white')
    
    #display(data)
    
    # difference heatmap plot
    model_names = data['model_name'].unique()
    if len(model_names)!=2:
        raise RuntimeWarning("The sub dataframe contains more than two models, which is not permissible for model comparison.")
        
    sub_data_m1 = data[data['model_name']==model_names[0]]
    sub_data_m2 = data[data['model_name']==model_names[1]]
    
    
    # ToDO
    if bootstrap:
        raise NotImplementedError("Bootstrapping is not implemented (yet) for diff heatmaps.")
    else:
        x_mean_m1 = pd.pivot_table(sub_data_m1, index=[index], columns=columns,
                                    values=values, aggfunc=np.mean)
        x_mean_m2 = pd.pivot_table(sub_data_m2, index=[index], columns=columns,
                                    values=values, aggfunc=np.mean)
        x_mean = x_mean_m1-x_mean_m2
        mask = x_mean.isnull()
        if annot_std:
            x_std = pd.pivot_table(data, index=[index], columns=columns,
                                   values=values, aggfunc=np.std)
            labels = x_mean.applymap(lambda x: annot_fmt.format(x)) + x_std.applymap(lambda x: annot_std_fmt.format(x))
            sns.heatmap(x_mean, cmap=cmap, mask=mask, annot=labels, fmt='', **kwargs)
        else:
            sns.heatmap(x_mean, cmap=cmap, mask=mask, annot=x_mean, 
                        fmt=annot_fmt[annot_fmt.find('{')+2:annot_fmt.find('}')], 
                        **kwargs)
    if row_entry:
        plt.title(f"{model_names[0]} - {model_names[1]}\n {row_title}: {row_entry}")
    else:
        plt.title(f"{model_names[0]} - {model_names[1]}")
        
    
def diff_heat_maps_by_weights(re_data, values, 
                              comparisons_by_model_name,
                              title=None, 
                              index='weight_account', columns='weight_systematicity',
                              row=None,
                              row_title=None,
                              annot_std=False, annot_fmt="{:2.0f}\n", annot_std_fmt=r'$\pm${:2.1f}', vmin=0, vmax=1,
                              output_dir=None, file_name=None, index_label=r'$\alpha_A$', columns_label=r'$\alpha_S$', bootstrap=False):
    # First we construe a mapping from the given tuples `comparisons_by_model_name`, which 
    # we will later use for facetgrid/sns.map_dataframe
    mapping_model_name2comp_type = dict()
    for m1, m2 in comparisons_by_model_name:
        mapping_model_name2comp_type[m1] = f'{m1}-{m2}'
        mapping_model_name2comp_type[m2] = f'{m1}-{m2}'
    re_data['comp_types']=re_data.apply(lambda x: mapping_model_name2comp_type[x['model_name']], axis=1)
    
    g = sns.FacetGrid(re_data, col='comp_types', row=row, height=5, aspect=1)
    if title:
        g.fig.suptitle(title, y=1.01)
    #mask = pd.pivot_table(re_data, index=[index], columns=columns,
    #                      values=values, aggfunc=np.mean).isnull()
    g.map_dataframe(diff_heatmap_plot, cbar=False, values=values, index=index, columns=columns,         
                    facet_row=row, row_title = row_title,
                    annot_std=annot_std, annot_fmt=annot_fmt, annot_std_fmt=annot_std_fmt, vmin=vmin, vmax=vmax, bootstrap=bootstrap)
    g.set_axis_labels(columns_label, index_label)
    #g.set_titles("{col_name}")
    if (file_name is not None) and (output_dir is not None):
        g.savefig(path.join(output_dir, file_name + '.pdf'), bbox_inches='tight')
        g.savefig(path.join(output_dir, file_name + '.png'), bbox_inches='tight')    
        
        
def bootstrap_std(col, n_resamples=1000):
    # We return the standard deviation of the bootstrap distribution as std
    bootstrap_res = spst.bootstrap((list(col),), np.mean, 
                              confidence_level=0.95,
                              random_state=1, 
                              method='percentile',
                              n_resamples=n_resamples)
    return bootstrap_res.standard_error

def bootstrap_mean(col, n_resamples=1000):
    bootstrap_res = spst.bootstrap((list(col),), np.mean, 
                              confidence_level=0.95,
                              random_state=1, 
                              method='percentile',
                              n_resamples=n_resamples)
    # We return the mean of the bootstrap distribution as a point estimate 
    # of the mean (since the bootstrap distribution is the mean for each 
    # bootstrap sample, it is a mean of a mean). I guess this makes more 
    # sense than to take the mean of the actual ensemble (as we did before).
    return np.mean(bootstrap_res.bootstrap_distribution)

    
def rel_share_of_property(re_data, 
                           property_col, 
                           col_rename = None,
                           groupby_cols=['model_short_name', 'model_name'],
                           collapse_branches=False,
                           cols_group_branches = ['model_name','ds','init_coms', 'weight_account', 'weight_systematicity'],
                           explode_cols = None,
                           bootstrap=False, n_resamples=1000):
    """
    This function uses single bool values or boolean-lists (in the column `property_col`) to calculate the share of these 
    properties. 
    
    
    :param bootstrap:
        If `bootstrap` is set to `True`, the relative share of True values in the subset defined by `groupby_cols` 
        will not be calculated by calculating the actual relative share in the subset, but by calculating the mean
        of a bootstrap distribution (of relative shares). Additinally, the standard deviation of the bootstrap 
        distribution will be added as an additional row (see <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html>
        for details).
    :param property_col: The name of the colums whose relative share we are interested. The entries should be single boolean values or lists of 
        boolean values.
    
    :param collapse: 
        If `True` rows will be reduced to one row per simulation setup (which assumes a corresponding
        redundancy in the column `property_col`).
    :param explode_cols:
        If not `None` the function will explode according to the passed cols. 
        Otherwise, it will will explode in `property_col` if (and only if) it contains list-like entries. 
    
    
    :return: 
    
    """
    ### Implementation idea:
    # 1. Reduce re_date to one row per simulation setup.
    # 2. Explode the table w.r.t. the col of interest (i.e., expand the list into rows).
    # 3. Aggregate the booleans of that col for the subsets of interest (`groupby_cols`)
    #    to calculate the relative share ('mean') and absolute values if wished.
    ###
    
    if col_rename is None:
        col_rename = {'mean': f'rel_{property_col}',
                      'std': f'std_{property_col}', 
                      'size': f'size_{property_col}', 
                      'sum': f'sum_{property_col}'}
    
    #### Collapse branches if wished ("result" perspective)
    # Cols that should have identical values for all branches (and only for those) that 
    # belong to one branching model run
    if collapse_branches:
        re_data = re_data.drop_duplicates(cols_group_branches)
    
    # specification of those cols (besides `properties_list_col`) that are relevant and later needed for the groupby
    # ToDo: We could simply use all cols.
    #relevant_cols = ["model_name", "model_short_name", "ds", "init_coms", "weight_account", "weight_systematicity"]
    #re_data = re_data[(relevant_cols + [property_col])]

    #### Explode in `property_col` if it contains lists
    # (e.g. in the column `go_coms_consistent`)
    # To-Do (?): Actually, we should check whether all entries have the same type.
    if explode_cols is not None:
        # To-Do: test this
        re_data = re_data.explode(explode_cols)
    elif isinstance(re_data[property_col].dropna().iloc[0], list):
        re_data = re_data.explode(property_col)
            
    # via `col_name` it can be indicated whether besides the relative share 
    # the absolute values should be added as cols as well: 
    if bootstrap:
        # ToDo: The will lead to a double bootstrapping. It would be better to bootstrap only once.
        agg_fun = [('bootstrap_mean', lambda x: bootstrap_mean(x, n_resamples=n_resamples)), 
                   ('bootstrap_std', lambda x: bootstrap_std(x, n_resamples=n_resamples))]
        # add the corresponding new colname
        col_rename['bootstrap_mean']=col_rename['mean']
        col_rename['bootstrap_std']=col_rename['std'] 
    else:
        # if we don't calculate stds we can take the actual mean and the the point estimated
        # provided by the bootstrap distribution (right?)
        # since we deal with a list of bools, `mean` will already return the relative share of `True`s in the respective subset 
        agg_fun = ['mean']
    if 'sum' in col_rename.keys():
        agg_fun.append('sum')
    if 'size' in col_rename.keys():
        agg_fun.append('size')
    result_df = re_data.groupby(groupby_cols)[property_col].agg(agg_fun).rename(columns=col_rename)
    return result_df

def set_errorbar_plot_style(n_hues=4, models=False):
    
    # set figure size
    plt.rcParams["figure.figsize"] = (9, 7.5)
    
    sns.set_theme(style="darkgrid", 
                  font_scale = 1.25,
                  rc={#"axes.spines.right": False, 
                      #"axes.spines.top": False,
                      #"axes.grid" : True,
                      #"axes.grid.axis": "y",
                      "lines.linewidth": 1.5})

    # set the default color cycle on basis of the color palette "coolwarm"
    if models:
        if n_hues == 4:
            # reorder four elements to fit linear-quadratic and global-local distinctions
            color_name_list = ['#6788ee', '#9abbff', '#e26952', '#f7a889']
        else:
            color_name_list = sns.color_palette("coolwarm", n_hues).as_hex()
    else:
        color_name_list = sns.color_palette("coolwarm", n_hues).as_hex()
    
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_name_list)
    
    
def set_heatmap_plot_style():
    
    # set figure size
    plt.rcParams["figure.figsize"] = (9, 7.5)
    
    sns.set_theme(style="darkgrid",
                  font_scale = 1.0)
    
def consistency_case_barplot(data, 
                             endpoint_type, 
                             init_coms_consistent, 
                             go_models = ["LinearGlobalRE","QuadraticGlobalRE"],
                             on_colab = True,
                             analyse_branches = False):
    
    # restrict global optima dataframe to two variants
    if endpoint_type=="go":
        rdata = data[data["model_name"].isin(go_models)]
    else:
        rdata=data
    
    if init_coms_consistent:
        y_cols = ["rel_{}_consistency_preserving_case".format(endpoint_type),
                  "rel_{}_consistency_eliminating_case".format(endpoint_type)]
    else:
        y_cols = ["rel_{}_inconsistency_eliminating_case".format(endpoint_type),
                  "rel_{}_inconsistency_preserving_case".format(endpoint_type)]    
    
    fig = px.bar(rdata.round(3), 
                  x="model_name", 
                  y=y_cols,
                  barmode="stack", 
                  text_auto=True)

    fig.update_yaxes(range=[0.0, 1.0])
    fig.update_traces(name="Consistency preserving (CP)", 
                       marker_color=coolwarm[0], 
                       selector=dict(name='rel_{}_consistency_preserving_case'.format(endpoint_type)))
    
    fig.update_traces(name="Inconsistency eliminating (IE)", 
                       marker_color=coolwarm[2],
                       selector=dict(name='rel_{}_inconsistency_eliminating_case'.format(endpoint_type)))
    
    fig.update_traces(name="Inconsistency preserving (IP)",
                       marker_color=coolwarm[3],
                       selector=dict(name='rel_{}_inconsistency_preserving_case'.format(endpoint_type)))
    
    fig.update_traces(name="Consistency eliminating (CE)",
                       marker_color=coolwarm[1], 
                       selector=dict(name="rel_{}_consistency_eliminating_case".format(endpoint_type)))

    fig.update_layout(template="plotly_white",
                       #paper_bgcolor="#e9e8e6",
                       #plot_bgcolor="#e9e8e6",
                       font={"color": "black", "size":12},
                       width=860 if endpoint_type=="fp" else 580,
                       margin={"t":80}
                      )

    fig.update_xaxes(title="Model variant", showticklabels=True, showgrid=False, linecolor= 'DarkGrey')
    fig.update_yaxes(title="Relative share", ticks="outside", tickcolor="DarkGrey", showgrid=False, linecolor='DarkGrey', zeroline=True, zerolinecolor="DarkGrey",
                      zerolinewidth=1)
    fig.update_layout(legend_title_text="")
    fig.update_layout(legend={'traceorder':'reversed', 
                              "orientation":"v", "x":1.0, "y":1.025, "xanchor":"left"})
    #fig.update_layout(title_text="Relative share of consistency cases among {}".format("global optima" if endpoint_type=="go" else "fixed points"))
    fig.update_traces(opacity=0.8)
    fig.show()



    if not on_colab:
        file_name = 'consistency_cases_{}_{}_{}.png'.format(endpoint_type,
                                                                 "ic_cons" if init_coms_consistent else "ic_incons",
                                                                 'pp' if analyse_branches else 'rp')
        fig.write_image(path.join(figures_output_dir, file_name), scale=2)
        
        file_name = 'consistency_cases_{}_{}_{}.pdf'.format(endpoint_type,
                                                                 "ic_cons" if init_coms_consistent else "ic_incons",
                                                                 'pp' if analyse_branches else 'rp')
        fig.write_image(path.join(figures_output_dir, file_name), scale=2)
        
def consistency_case_heatmaps_by_weights(data, 
                             endpoint_type, 
                             case_name,
                             go_models = ["LinearGlobalRE","QuadraticGlobalRE"],
                             analyse_branches = False,
                             bootstrap = False,
                             n_resamples = 400):
     
    # restrict global optima dataframe to two variants
    if endpoint_type=="go":
        rdata = data[data["model_name"].isin(go_models)]
    else:
        rdata=data
        
    if case_name in ["consistency_preserving", "consistency_eliminating"]:
        # consistent init coms
        rdata = rdata[rdata["init_coms_dia_consistent"]]
    else:
        # inconsistent init coms
        rdata = rdata[~rdata["init_coms_dia_consistent"]]
         
    
    values_name = endpoint_type + '_' + case_name + '_case' 
        
    
    display_case_name = " ".join(case_name.split("_"))
    
    heat_maps_by_weights(re_data = rdata, 
                     values = values_name, 
                     #title = '{} share of {} cases for {}{}'.format(metric, display_case_name, display_endpoint_type, branches), 
                     annot_std = False,
                     annot_fmt="{:2.2f}\n", 
                     annot_std_fmt = r'$\pm${:2.2f}',
                     vmin=0, vmax=1,
                     bootstrap = bootstrap,
                     n_resamples = n_resamples)
    

def get_lengths(x, pos_type):
    # working on strings directly in order to avoid having to literal_eval millions of strings
    
    # split the list of endpoint tuples 
    l = x.split('), (')

    # split each endpoint tuple into a theory and a commitment position (set representation with {})
    l = [s.split('}, {') for s in l]
    
    if not pos_type in ["theory", "commitments"]:
        raise ValueError("pos_type must be 'theory' or 'commitments'")
    
    idx = 0 if pos_type == "theory" else 1
     
    # length of the first (theory) or second (commitments) element  after 
    # the comma-separated sentences have been split
    return [len(s[idx].split(',')) for s in l]


######################################################################    
##################### Alternative Systematicity Measures #############
######################################################################    
    
# helper functions
gg = lambda x: 1 - x**2

def sys_standard(theory_size, theory_clos_size):
    return gg((theory_size-1)/theory_clos_size)

def sys_pure_simpl(theory_size, n):
    return gg((theory_size-1)/n)

def sys_af(theory_size, sig_theory, n):
    return gg((theory_size*(log(sig_theory, 2)+1))/(n*(n+1)))


def restricted_sigma(tau: DialecticalStructure, domain: Position):
    return len([pos for pos in tau.consistent_positions() if pos.domain()==domain])

def conditional_restricted_sigma(tau: DialecticalStructure, domain: Position, condition: Position):
    n = tau.sentence_pool().size()
    # If the given `domain` is the whole domain minus the condition's domain, the conditional
    # restricted sigma is simply sigma given the condition:
    restricted_domain = StandardPosition.from_set(tau.sentence_pool().domain().as_set() - condition.domain().as_set(), n)
    if restricted_domain == domain:
        return tau.n_complete_extensions(condition)
    
    pos_sigma_conditional_restricted = {StandardPosition.from_set(pos.as_set() & domain.as_set(), n) for 
                                        pos in 
                                        tau.consistent_complete_positions() if 
                                        condition.is_subposition(pos)}
    return len(pos_sigma_conditional_restricted)

def sys_global(tau: DialecticalStructure, theory: Position):
    n = tau.sentence_pool().size()
    restricted_domain = StandardPosition.from_set(tau.sentence_pool().domain().as_set() - theory.domain().as_set(), n)
    # bug: restricted_domain.size() 
    restricted_domain_size = len(restricted_domain.as_set()) 
    
    if restricted_domain.size() == 0:
        return 0
    else:
        return (log(restricted_sigma(tau,restricted_domain),2)-log(conditional_restricted_sigma(tau, restricted_domain, theory),2))/(restricted_domain_size/2)


def sys_sigligs(tau: DialecticalStructure, theory: Position):
    n = tau.sentence_pool().size()
    restricted_domain = StandardPosition.from_set(tau.sentence_pool().domain().as_set() - theory.domain().as_set(), n)
    # bug: restricted_domain.size() 
    restricted_domain_size = len(restricted_domain.as_set()) 
    
    if restricted_domain.size() == 0:
        return 0
    else:
        #sig = tau.n_complete_extensions()
        return (log(restricted_sigma(tau,restricted_domain),2)-log(conditional_restricted_sigma(tau, restricted_domain, theory),2))/(n-1)

def sys_sigligs_2(tau: DialecticalStructure, theory: Position, denom: float):
    n = tau.sentence_pool().size()
    restricted_domain = StandardPosition.from_set(tau.sentence_pool().domain().as_set() - theory.domain().as_set(), n)
    # bug: restricted_domain.size() 
    restricted_domain_size = len(restricted_domain.as_set()) 
    
    if restricted_domain.size() == 0:
        return 0
    else:
        sig = tau.n_complete_extensions()
        return (log(restricted_sigma(tau,restricted_domain),2)-log(conditional_restricted_sigma(tau, restricted_domain, theory),2))/denom
        #return (log(restricted_sigma(tau,restricted_domain),2)-log(conditional_restricted_sigma(tau, restricted_domain, theory),2))/(n-1)

    
def sys_sigquags(tau: DialecticalStructure, theory: Position):
    return 1-(1-sys_sigligs(tau, theory))**2

def sligs_a(theory_size, theory_clos_size, n, alpha):
    # penalties
    score = alpha * (theory_size - 1) + (1 - alpha) * (n - theory_clos_size)
    # denominator "c" for normalisation
    denom = theory_clos_size * (2*alpha-1) + n * (1-alpha) - alpha # (n-1)*0.5 + abs(alpha-0.5) * (n-1)
    # linear G
    return 1-score/denom

def s_sligs_b(theory_size, theory_clos_size, n, beta):
    # penalties
    score = beta * (theory_size - 1) + (1 - beta) * (n - theory_clos_size)
    # alternative denominator for normalisation
    denom = (abs(beta - 0.5) + 0.5) * (n - 1)
    # linear G
    return 1-score/denom