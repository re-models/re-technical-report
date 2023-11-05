
from ast import literal_eval
from os import getcwd, path
from pathlib import Path
import tarfile

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from typing import Set, List

def literal_eval_cols(data: DataFrame, cols: List[str]):
    for col_name in cols:
        data[col_name] = data.apply(lambda x: literal_eval(x[col_name]), axis=1)

# replace string 'set()' for literal_eval to work  properly
def replace_set(row, label):
    return row[label].replace('set()', '{}')

def load_re_data(data_dir, data_file_name):
    if data_file_name[data_file_name.find('.'):len(data_file_name)] == '.csv.tar.gz':
        with tarfile.open(path.join(data_dir,data_file_name)) as tar:
            for tarinfo in tar:
                file_name = tarinfo.name
            tar.extractall(data_dir)
        re_data = pd.read_csv(path.join(data_dir, file_name))
    else:
        re_data = pd.read_csv(path.join(data_dir,data_file_name))


    re_data['global_optima'] = re_data['global_optima'].replace('set()', '{}')
    re_data['fixed_points'] = re_data['fixed_points'].replace('set()', '{}')

    #re_data['fixed_points'] = re_data.apply(lambda row: replace_set(row, 'fixed_points'), axis=1)

    # WARNING: XXX (eval not being save)
    # converting mere strings to data-objects
    literal_eval_cols(re_data, ['global_optima',
                                'go_coms_consistent',
                                'go_union_consistent',
                                'go_full_re_state',
                                'go_fixed_point',
                                'go_account',
                                'go_faithfulness',
                                'fixed_points',
                                'fp_coms_consistent',
                                'fp_union_consistent',
                                'fp_full_re_state',
                                'fp_account',
                                'fp_faithfulness',
                                'fp_global_optimum',
                               ])
    return re_data

