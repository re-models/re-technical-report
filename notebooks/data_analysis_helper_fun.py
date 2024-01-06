
from ast import literal_eval
from os import getcwd, path
from pathlib import Path
import tarfile

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Set, List
import random
import scipy.stats as spst
from itertools import combinations


from matplotlib_venn import venn2


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
    
    set_errorbar_plot_style()
    
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

def set_errorbar_plot_style():
    
    sns.set_theme(style="darkgrid", 
                  font_scale = 1.5,
                  rc={#"axes.spines.right": False, 
                      #"axes.spines.top": False,
                      #"axes.grid" : True,
                      #"axes.grid.axis": "y",
                      "lines.linewidth": 2.5})

    # set the default color cycle on basis of the color palette "coolwarm"
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#6788ee', '#9abbff', '#e26952', '#f7a889'])
    
def set_heatmap_plot_style():
    sns.set_theme(style="darkgrid",
                  font_scale = 1.0)
