# BreastCancerSpatialProteomics

## Data description 

`DSP_protein_discovery_cohort.txt` - contains DSP protein counts used to generate Figures 1c-d, 2, 3, and 4.

`bulk_RNA_data.txt` - contains the bulk RNA expression data used for model comparison in Figure 4.

`DSP_protien_validation_cohort.txt` - contains the DSP protein counts for the 29 cases  in the text cohort used to generate Figure 5 and corresponding Extended Data 9.

`perimetric_complexity.txt` - contains the perimetric complexity values per region used for Extended Data 8. 

`cd45_R_data.txt` - contain the on-treatment CD45 IHC and DSP values using to compare single feature CD45 IHC and DSP in Figure 6.  

## Code description

`volcano_waterfall.R` is an R script with example functions used to run the linear mixed-effect models and generate the volcano plots and waterfall plots shown in Figures 2a, 2b, 2d, 2e as well as the Extended Data 1,3,4,5,7, and 9.

`classifier.py` is a Python3 script used for model comparisons and evaluation of performance via internal cross-validation in Figure 4.

`classifier_test.py` is used for evaluation of model performance in an independent validation cohort as shown in Figure 5. 

`DSP_IHC_comparison.ipynb` is a Python3-Jupyter script used for evaluation of the single feature CD45 IHC and DSP models. 
