# BreastCancerSpatialProteomics

## Data description 

`DSP_protein_discovery_cohort.txt` - contains DSP protein counts used to generate Figures 1c-d, 2, 3, and 4.

`bulk_RNA_data.txt` - contains the bulk RNA expression data used for model comparison in Figure 4.

`DSP_protien_validation_cohort.txt` - contains the DSP protein counts for the 29 cases  in the text cohort used to generate Figure 5 and corresponding Extended Data 9.

`perimetric_complexity.txt` - contains the perimetric complexity values per region used for Extended Data 8. 

`cd45_R_data.txt` - contain the on-treatment CD45 IHC and DSP values using to compare single feature CD45 IHC and DSP in Figure 6.  

## Code description

`volcano_waterfall.R` is an R v3.6.0 cript with example functions used to run the linear mixed-effect models and generate the volcano plots and waterfall plots shown in Figures 2a, 2b, 2d, 2e as well as the Extended Data 1,3,4,5,7, and 9.

`classifier.py` is a Python v3.7.4 script used for model comparisons and evaluation of performance via internal cross-validation in Figure 4.

`classifier_test.py` is used for evaluation of model performance in an independent validation cohort as shown in Figure 5. 

`DSP_IHC_comparison.ipynb` is a Python v3.7.4 Jupyter Notebook used for evaluation of the single feature CD45 IHC and DSP models.



## Steps to run the code

To run the code please first make sure that you have miniconda (https://docs.conda.io/en/latest/miniconda.html) or conda (https://docs.conda.io/) installed.


### Install require softwares

Next step is to create a conda env `bcsp` and install `Python v3.7.4`, `R v3.6.0` and required packages using the following command.

``` bash
conda create --name bcsp -c conda-forge -c conda-forge -c r python=3.7.4 jupyter pandas=0.25.1 numpy=1.17.2 scipy=1.3.1 scikit-learn=0.21.3 pystan=2.19.1.1 seaborn=0.9.0 statsmodels=0.10.1 arviz=0.10.0 matplotlib=3.1.2 blackcellmagic r-base=3.6.0 r-vioplot=0.3.2 r-zoo=1.8-6 r-sm=2.2-5.6 r-ggrepel=0.8.1 r-ggplot2=3.3.0 r-reshape=0.8.8 r-tidyr=1.0.3 r-lmerTest=3.1-0 r-lme4=1.1-21 r-matrix=1.2-17 r-dplyr=0.8.5 r-ggeffects
```

### Activate the conda env using this command

``` bash
conda activate bcsp
```

### Now clone the git repo

``` bash
git clone https://github.com/cancersysbio/BreastCancerSpatialProteomics.git BCSP && cd BCSP
```

### And run Jupyter Notebook using
``` bash
jupeter notebook
```
And locatet the `DSP_IHC_comparison.ipynb` Notebook in the Jupyter browser.

### Other scripts can be run as follows:

``` bash
Rscript volcano_waterfall.R
```
``` bash
python classifier.py
```
``` bash
classifier_test.py
```


