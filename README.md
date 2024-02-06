# Formal Models of Reflective Equilibrium - Technical Report

This repository contains the technical report ["Assessing a Formal Model of Reflective Equilibrium"](https://re-models.github.io/re-technical-report/) for the SNSF-DFG project ["Formal Models of Reflective Equilibrium - How Far Does Reflective Equilibrium Take Us?"](https://www.philosophie.unibe.ch/forschung/forschungsprojekte/how_far_does_reflective_equilibrium_take_us/project/index_ger.html).

This repository has the following structure:

+ `data`: Contains all data files on which this report is based.
+ `notebooks`: [Jupyter notebooks](https://jupyter.org/) to reproduce the data analysis (see below).
+ `report`: [Quarto](https://quarto.org/) source files for this report.
+ `output`: Quarto generated output.
+ `re-technical-report`: A small Python package that provides some helper functions for data generation and analysis.

You can read this report in the browser (<https://re-models.github.io/re-technical-report/>) or as pdf file (XX)

## Reproducing the Results

All findings and the underlying data can be reproduced by using the [Python implementation](https://github.com/re-models/rethon) of the model. For each chapter you will find [here](https://github.com/re-models/re-technical-report/tree/main/notebooks) a Jupyter notebook whose execution produces all analysis results. You can execute them on [Colab](https://colab.research.google.com/) or locally by cloning this repository.

### Using Colab

Simply click the Colab badge (![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)) in the notebooks. Note, however, that it is not possible to use the complete data set on Colab. Instead, you are confined to using the subdata set `TINY` (see below).  

### Running the Notebooks locally

You can execute the notebooks locally by proceeding along the following steps:

1. Install python together with JupyterLab (<https://jupyter.org/install>)
2. Git-clone the repository (with `git clone git@github.com:re-models/re-technical-report.git`).
3. Open notebooks in JupyterLab and execute the cells.

By default, the notebooks use a convenience function to import one of the three data sets by specifiying the data set's name (e.g., `re_data_by_name(data_name="TINY")`). The function handles data retrieval in dependence of whether the notebook is executed on Colab or locally. 

Alternatively, you can load the data by specifying the directory and file name of the data file in the following way:

```python
re_data = load_re_data(data_dir=path.join(Path(getcwd()).parent.absolute(), 'data'), 
                       data_file_name='re_data_tau_alpha_mini.tar.gz')
```


## Data

The data that the model produced can be found [here](https://github.com/re-models/re-technical-report/tree/main/data). We provide three different data sets:

1. The `FULL` data set (file name `re_data_tau_alpha.tar.gz`, extracted $\sim 12GB$): The full data set that is described and used in the report (👉 <https://re-models.github.io/re-technical-report/intro.html#ensemble-description>)
2. The `MINI` data set (file name `re_data_tau_alpha_mini.tar.gz`, extracted $\sim 800MB$): A subset of the full data set based on $20$ dialectical structures.
3. The `TINY` data set (file name `re_data_tau_alpha_tiny.csv`, $\sim 12MB$): A subset of the full data set based on $2$ dialectical structures.

Note that using the full data set might, depending on the available RAM, overload your computational resources.  

## Credits

Earlier versions of this report were discussed on several occasions with all members of the research project  ['How far does Reflective Equilibrium Take us? Investigating the Power of a Philosophical Method'](https://www.philosophie.unibe.ch/forschung/forschungsprojekte/how_far_does_reflective_equilibrium_take_us/project/index_ger.html) (SNSF grant 182854 and German Research Foundation grant 412679086). We thank, in particular, Claus Beisbart, Gregor Betz, Georg Brun, Alexander Koch and Richard Lohse for their helpful comments, which helped to improve this report considerably.

## Citing

xxx

---
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
