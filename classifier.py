#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as ss
# import cufflinks as cf
import pickle
import pystan
import arviz as az
# cf.go_offline()
import os
from IPython import get_ipython
from itertools import combinations
get_ipython().run_line_magic('reload_ext', 'blackcellmagic')
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    cross_val_score,
    cross_validate,
    LeaveOneOut,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    permutation_test_score
)
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

# set a random state
# also need to modify number of job in the cross_validate function, depending on your compute system
random_state = 

# read in the DSP data file
# file must contain a column called pCR, ER and pam50 status as well as the prootein markers of interest 
#(see markers list below)
dat_f = "cd45_R_data.txt"

raw_df = pd.read_csv(dat_f, sep="\t", index_col=0)
raw_df = raw_df.rename(mapper=lambda x: x.replace(".", "_"), axis=1)
markers = [
    "Beta.Catenin",
    "pS6",
    "PTEN",
    "P.ERK",
    "S6",
    "Ki.67",
    "Beta.2.microglobulin",
    "AKT",
    "PSTAT3",
    "p.AKT",
    "Her2",
    "Pan.Cytokeratin",
    "CD8",
    "B7.H3",
    "CD4",
    "CD68",
    "CD14",
    "GZMB",
    "CD3",
    "CD66B",
    "VISTA",
    "PD1",
    "CD44",
    "CD56",
    "PD.L1",
    "CD45",
    "CD19",
    "CD45RO",
    "Bcl.2",
    "B7.H4.VTCN1",
    "STING.TMEM173",
    "IDO.1",
    "Lag3",
    "CD11c",
    "ICOS.CD278",
    "OX40L.CD252.TXGP1",
    "CD27",
    "CD163",
    "FOXP3",
    "X4.1BB"
]
markers = list(map(lambda x: x.replace(".", "_"), markers))
tumor_markers = [
    "Beta.Catenin",
    "pS6",
    "PTEN",
    "P.ERK",
    "S6",
    "Ki.67",
    "AKT",
    "PSTAT3",
    "p.AKT",
    "Her2",
    "Pan.Cytokeratin",
]
tumor_markers = list(map(lambda x: x.replace(".", "_"), tumor_markers))
immune_markers = [
    "CD8",
    "CD4",
    "CD68",
    "CD14",
    "GZMB",
    "CD3",
    "CD66B",
    "VISTA",
    "PD1",
    "CD44",
    "CD56",
    "CD45",
    "CD19",
    "CD45RO",
    "STING.TMEM173",
    "Lag3",
    "CD11c",
    "ICOS.CD278",
    "CD27",
    "CD163",
    "B7.H3",
    "PD.L1",
    "B7.H4.VTCN1",
    "IDO.1",
    "OX40L.CD252.TXGP1",
    "FOXP3",
    "X4.1BB"
]
immune_markers = list(map(lambda x: x.replace(".", "_"), immune_markers))
general_markers = ["Beta.2.microglobulin", "Bcl.2"]
general_markers = list(map(lambda x: x.replace(".", "_"), general_markers))

# Collapse multiple tissue samples for a patient timepoint into their mean and take B and R timepoints
patient_timepoint_means = raw_df.loc[raw_df.timepoint.isin(["B", "R"]), markers + igg + ["patient", "timepoint"]].groupby(["patient", "timepoint"]).mean().rename(lambda x: x+"_mean", axis=1)

# Get complete cases only (must have both B and R timepoints)
complete_patient_timepoint_means = patient_timepoint_means.loc[
    patient_timepoint_means.index.levels[0][
        patient_timepoint_means.loc[pd.IndexSlice[:, ["B", "R"]], :]
        .reset_index("timepoint")
        .groupby("patient", sort=False)["timepoint"]
        .nunique()
        == 2
    ]
]

# Calculate RminusB
RminusB = complete_patient_timepoint_means.loc[pd.IndexSlice[:, "R"], :].droplevel(axis=0, level=1)  - complete_patient_timepoint_means.loc[pd.IndexSlice[:, "B"], :].droplevel(axis=0, level=1)
RminusB["timepoint"] = "RminusB"

# Add to dataframe
complete_patient_timepoint_means = pd.concat([complete_patient_timepoint_means.reset_index(level=1), RminusB], axis=0)
# Reshape to wide format
complete_patient_timepoint_means = complete_patient_timepoint_means.reset_index().pivot_table(
    index="patient", values=map(lambda x: x + "_mean", markers + igg), columns="timepoint"
)

# Add on the response data
complete_patient_timepoint_means["pCR"] = raw_df.groupby("patient")["pCR"].first()
# Add on ER and pam50

ERstatus = raw_df.groupby("patient")[["ER"]].first()
pam50status = raw_df.groupby("patient")["pam50"].first().fillna("other") # pam50 is unknown for one patient, fill it with "other"
complete_patient_timepoint_means["ER_mean", "B"] = ERstatus # pardon the mean naming, makes code simpler
complete_patient_timepoint_means["pam50_mean", "B"] = pam50status.astype("category").cat.codes # pardon the mean naming, makes code simpler
# # Should not have any null values
assert complete_patient_timepoint_means.isnull().any().any() == False


# log loss scorer for leave one out - must set the possible label lavels
log_loss_with_labels = make_scorer(lambda y_true, y_pred: log_loss(y_true, y_pred, labels=(0, 1)), needs_proba=True, greater_is_better=False)

class FeatureSelector(TransformerMixin, BaseEstimator):
    """
    Subclass to select features for introduction into pipeline.
    Setup to answer questions above 
    Select features based on timepoints, change in timepoints, and different markers
    """

    def __init__(self, marker_names, timepoints):
        """
        :param list names: one or more marker names
        :param list timepoints: one or more of ["B", "R", "RminusB"]
        """

        super().__init__()

        self.marker_names = marker_names
        self.timepoints = timepoints

    def transform(self, X):
        """
        X is assumed to have multilevel columns, marker names at level 0 and marker timepoints at level 1
        """
        return X.loc[:, pd.IndexSlice[self.marker_names, self.timepoints]]

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return X.loc[:, pd.IndexSlice[self.marker_names, self.timepoints]]

# The folds to use in the inner CV loop used by LogisticRegressionCV
kf = StratifiedKFold(n_splits=5, random_state=100, shuffle=True)

def run_cross_validation(dataframe, features_to_test, scoring_names):
    """
    Run cross validation and return scores
    :param pd.DataFrame dataframe: a wide dataframe, with single patient per row, multiindex columns w/ level 0 marker, level 1 timepoint. also must contain "pCR" as a column
    :param dict features_to_test: dictionary with label as keys, and tuple as values. First index of tuple is markers, Second index of tuple is timepoint
    :param list scoring_names: list of sklearn metrics to use for scoring
    :return pd.DataFrame scores: dataframe of scores
    """
    
    pipelines_to_compare = dict(
        [
            (
                k,
                make_pipeline(
                    FeatureSelector(*v),
                    StandardScaler(),
                    LogisticRegressionCV(
                        Cs=100,
                        penalty="l2",
                        solver="liblinear",
                        class_weight="balanced",
                        scoring="accuracy",
                        random_state=random_state,
                        cv=kf,
                    )
                ),
            )
            for k, v in features_to_test.items()
        ]
    )
    ## 100x repeated max-fold stratified cross-validation, then unpaired t-testing for significance
    scores = {}

    for feature, pipeline in pipelines_to_compare.items():
        scores[feature] = pd.DataFrame.from_records(
            cross_validate(
                pipeline,
                dataframe,
                dataframe.pCR,
                cv=RepeatedStratifiedKFold(random_state=random_state,
                    n_splits=dataframe.pCR.value_counts().min()//2, # max-folds for stratified is the minimum of number of samples in class
                    n_repeats=100 
                ),
                scoring=scoring_names,
                n_jobs=36, # modify number of jobs as needed
            )
        )
    scores = pd.concat(scores.values(), keys=scores.keys(), names=["marker_group", "fold"])
    
    return scores

def get_model_weights(dataframe, features_to_use):
    """
    Train a model and then return model weights
    :param pd.DataFrame dataframe:  a wide dataframe, with single patient per row, multiindex columns w/ level 0 marker, level 1 timepoint. also must contain "pCR" as a column
    :param tuple features_to_use: Tuple of features to use. First index of tuple is markers, Second index of tuple is timepoint
    :return pd.DataFrame weights: dataframe of model intercept and model coefficients with feature names as columns
    """
    
    model_pipeline = make_pipeline(
                    FeatureSelector(*features_to_use),
                    StandardScaler(),
                    LogisticRegressionCV(
                        Cs=100,
                        penalty="l2",
                        solver="liblinear",
                        class_weight="balanced",
                        scoring="accuracy",
                        random_state=random_state,
                        cv=kf,
                    )
                )
    model_pipeline.fit(dataframe, dataframe.pCR)
    
    weights = pd.DataFrame(model_pipeline[-1].coef_.T, index=dataframe.loc[:, pd.IndexSlice[features_to_use[0], features_to_use[1]]].columns, columns=["weight"])
    weights = weights.append(pd.DataFrame([[model_pipeline[-1].intercept_[0]]], index=pd.MultiIndex.from_tuples([("intercept", "NA")]), columns=["weight"])).rename_axis(["marker", "timepoint"])
    return weights
    

def calculate_significance(scores):
    """
    Calculates significance with unpaired t-test and holm-bonferroni correction
    :param pd.DataFrame scores: dataframe output by sklearn cross_validate function
    """
    
    # significance testing
    cc = [list(a) for a in combinations(scores.index.levels[0],2)]
    metric_p_values = {}
    for metric in scores.columns:
        p_values = {}
        for combination in cc:
            p_values[tuple(combination)] = ss.ttest_ind(scores.loc[combination[0], metric], scores.loc[combination[1], metric])[1]

        metric_p_values[metric] = p_values
    metric_p_values = pd.DataFrame.from_records(metric_p_values)
    significances = pd.concat([
        metric_p_values,
        metric_p_values.apply(lambda x: sm.stats.multipletests(x, alpha=0.05, method="holm")[1], axis=0)
    ], axis=1, keys=["p-value", "adj_p-value"])
    return significances

def do_permutation_test_vs_shuffled_null(dataframe, features_to_use):
    """
    Train a model and compare using permutation test to null
    :param pd.DataFrame dataframe:  a wide dataframe, with single patient per row, multiindex columns w/ level 0 marker, level 1 timepoint. also must contain "pCR" as a column
    :param tuple features_to_use: Tuple of features to use. First index of tuple is markers, Second index of tuple is timepoint
    :return The true score without permuting targets, The scores obtained for each permutations, The p-value, which approximates the probability that the score would be obtained by chance. 
    """
    
    model_pipeline = make_pipeline(
                    FeatureSelector(*features_to_use),
                    StandardScaler(),
                    LogisticRegressionCV(
                        Cs=100,
                        penalty="l2",
                        solver="liblinear",
                        class_weight="balanced",
                        scoring="accuracy",
                        random_state=random_state,
                        cv=kf,
                    )
                )
    score, permutation_scores, pvalue = permutation_test_score(model_pipeline, dataframe, dataframe.pCR, scoring="roc_auc", cv=RepeatedStratifiedKFold(random_state=random_state,n_repeats=10,n_splits=dataframe.pCR.value_counts().min()//2), n_permutations=1000, n_jobs=36, )
    
    return score, permutation_scores, pvalue


# In[17]:


def run_cross_validation_plots(dataframe, features_to_test):
    """
    Run cross validation and return scores for plotting
    :param pd.DataFrame dataframe: a wide dataframe, with single patient per row, multiindex columns w/ level 0 marker, level 1 timepoint. also must contain "pCR" as a column
    :param dict features_to_test: dictionary with label as keys, and tuple as values. First index of tuple is markers, Second index of tuple is timepoint
    :return tuple (pd.DataFrame roc_scores, pd.DataFrame pr_scores)
    """
    pipelines_to_compare = dict(
        [
            (
                k,
                make_pipeline(
                    FeatureSelector(*v),
                    StandardScaler(),
                    LogisticRegressionCV(
                        Cs=100,
                        penalty="l2",
                        solver="liblinear",
                        class_weight="balanced",
                        scoring="accuracy",
                        random_state=random_state,
                        cv=kf,
                    ),
                ),
            )
            for k, v in features_to_test.items()
        ]
    )
    ## 100x repeated max-fold stratified cross-validation, then unpaired t-testing for significance
    roc_scores = {}  # used to store fpr and tpr
    pr_scores = {}

    def train_and_score_model(pipeline, train_X, test_X, train_y, test_y):
        pipeline.fit(train_X, train_y)
        preds = pipeline.decision_function(test_X)
        fpr, tpr, roc_threshs = roc_curve(test_y, preds)
        precision, recall, pr_threshs = precision_recall_curve(test_y, preds)
        return (
            pd.DataFrame([fpr, tpr], index=["fpr", "tpr"], columns=roc_threshs),
            pd.DataFrame(
                [recall, precision],
                index=["recall", "precision"],
                columns=list(pr_threshs) + [1000],
            ),
        )
    
    for feature, pipeline in pipelines_to_compare.items():
        rskf = RepeatedStratifiedKFold(
            random_state=random_state,
            n_splits=dataframe.pCR.value_counts().min()
            // 2,  # max-folds for stratified is the minimum of number of samples in class, but we want more than one positive/negative class in each test fold
            n_repeats=100,
        )
        pr_data = []
        data = Parallel(n_jobs=32)(
            delayed(train_and_score_model)(
                pipeline,
                dataframe.iloc[train],
                dataframe.iloc[test],
                dataframe.pCR.iloc[train],
                dataframe.pCR.iloc[test],
            )
            for train, test in rskf.split(dataframe, dataframe.pCR)
        )
        roc_data, pr_data = list(zip(*data))
        roc_scores[feature] = pd.concat(
            roc_data, keys=np.arange(len(roc_data)), names=["fold", "metric"]
        )
        pr_scores[feature] = pd.concat(
            pr_data, keys=np.arange(len(pr_data)), names=["fold", "metric"]
        )
    roc_scores = pd.concat(roc_scores, keys=features_to_test, names=["feature"])
    pr_scores = pd.concat(pr_scores, keys=features_to_test, names=["feature"])
    return roc_scores, pr_scores

def plot_roc(roc_scores):
    f, ax = plt.subplots()
    roc_scores = roc_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))
    ax.step(
        roc_scores.loc[pd.IndexSlice["B", "fpr", "mean"], :],
        roc_scores.loc[pd.IndexSlice["B", "tpr", "mean"], :],
        label="Mean Baseline (Mean AUC=x)",
    )
    ax.step(
        roc_scores.loc[pd.IndexSlice["R", "fpr", "mean"], :],
        roc_scores.loc[pd.IndexSlice["R", "tpr", "mean"], :],
        label="Mean Runin (Mean AUC=x)",
    )
    ax.step(
        roc_scores.loc[pd.IndexSlice["BandR", "fpr", "mean"], :],
        roc_scores.loc[pd.IndexSlice["BandR", "tpr", "mean"], :],
        label="Mean Baseline and Runin (Mean AUC=x)",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random Chance", alpha=0.8)
    ax.legend(bbox_to_anchor=(1.01, 1))
    ax.set_xlim([0, 1])
    ax.set_xlabel("fpr (1-specificity)")
    ax.set_ylim([0, 1])
    ax.set_ylabel("tpr (sensitivity)")
    f.suptitle("ROC")
    
def plot_pr(pr_scores):
    f, ax = plt.subplots()
    pr_scores = pr_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))
    ax.step(
        pr_scores.loc[pd.IndexSlice["B", "recall", "mean"], :],
        pr_scores.loc[pd.IndexSlice["B", "precision", "mean"], :],
        label="Mean Baseline (Mean AUC=x)",
    )
    ax.step(
        pr_scores.loc[pd.IndexSlice["R", "recall", "mean"], :],
        pr_scores.loc[pd.IndexSlice["R", "precision", "mean"], :],
        label="Mean Runin (Mean AUC=x)",
    )
    ax.step(
        pr_scores.loc[pd.IndexSlice["BandR", "recall", "mean"], :],
        pr_scores.loc[pd.IndexSlice["BandR", "precision", "mean"], :],
        label="Mean Baseline and Runin (Mean AUC=x)",
    )
    ax.legend(bbox_to_anchor=(1.01, 1))
    ax.set_xlim([0, 1])
    ax.set_xlabel("recall (sensitivity)")
    ax.set_ylim([0, 1])
    ax.set_ylabel("precision (ppv)")
    f.suptitle("Precision-Recall")

scoring_names = [
    "neg_log_loss",
    "accuracy",
    "brier_score_loss",
    "roc_auc",
    "average_precision",
]


# # Which timepoint or change between timepoints is most predictive? - Using all markers
features_to_test = {
    "B": (list(map(lambda x: x+"_mean", markers)), "B"),
    "R": (list(map(lambda x: x+"_mean", markers)), "R"),
    "BandR": (list(map(lambda x: x+"_mean", markers)), ["B", "R"])
}

scores = run_cross_validation(complete_patient_timepoint_means, features_to_test, scoring_names)

# summary statistics
summary_scores = scores.groupby(level=0).describe()
summary_scores.loc[:, pd.IndexSlice[map(lambda x: "test_" + x, scoring_names), ["mean", "std", "50%"]]]

roc_scores, pr_scores = run_cross_validation_plots(complete_patient_timepoint_means, features_to_test)
roc_scores = roc_scores.fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
pr_scores = pr_scores.fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
roc_scores = roc_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))
pr_scores = pr_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))

# plot roc
f, ax = plt.subplots()
ax.step(roc_scores.loc[pd.IndexSlice["B", "fpr", "mean"], :],
        roc_scores.loc[pd.IndexSlice["B", "tpr", "mean"], :],
        label="Pre-treatment (Mean AUC=xxx)",)
ax.step(roc_scores.loc[pd.IndexSlice["R", "fpr", "mean"], :],
        roc_scores.loc[pd.IndexSlice["R", "tpr", "mean"], :],
        label="On-treatment (Mean AUC=xxx)",)
ax.step(roc_scores.loc[pd.IndexSlice["BandR", "fpr", "mean"], :],
        roc_scores.loc[pd.IndexSlice["BandR", "tpr", "mean"], :],
        label="On-/Pre-treatment (Mean AUC=xxx)",)
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random Chance", alpha=0.8)
ax.legend(bbox_to_anchor=(1.01, 1))
ax.set_xlim([0, 1])
ax.set_xlabel("fpr (1-specificity)")
ax.set_ylim([0, 1])
ax.set_ylabel("tpr (sensitivity)")
f.suptitle("ROC")

# plot pr
f, ax = plt.subplots()
ax.step(pr_scores.loc[pd.IndexSlice["B", "recall", "mean"], :],
        pr_scores.loc[pd.IndexSlice["B", "precision", "mean"], :],
        label="Pre-treatment (Average Precision= xxx)",)
ax.step(pr_scores.loc[pd.IndexSlice["R", "recall", "mean"], :],
        pr_scores.loc[pd.IndexSlice["R", "precision", "mean"], :],
        label="On-treatment (Average Precision= xxx)",)
ax.step(pr_scores.loc[pd.IndexSlice["BandR", "recall", "mean"], :],
        pr_scores.loc[pd.IndexSlice["BandR", "precision", "mean"], :],
        label="On-/Pre-treatment(Average Precision=xxx)",)
ax.legend(bbox_to_anchor=(1.01, 1))
ax.set_xlim([0, 1])
ax.set_xlabel("recall (sensitivity)")
ax.set_ylim([0, 1])
ax.set_ylabel("precision (ppv)")
f.suptitle("Precision-Recall")

calculate_significance(scores).round(4)

# Retrain best model on entire dataset and get coefficients
best_model_weights = get_model_weights(complete_patient_timepoint_means, (list(map(lambda x: x+"_mean", markers)), ["B", "R"]))
best_model_weights.plot(kind="bar", figsize=(20,5))

# are immune or tumor markers more predictive?
features_to_test = {
    "markers": (list(map(lambda x: x+"_mean", markers)), ["B", "R"]),
    "tumor_markers": (list(map(lambda x: x+"_mean", tumor_markers)), ["B", "R"]),
    "immune_markers": (list(map(lambda x: x+"_mean", immune_markers)), ["B", "R"]),
}
scoring_names = [
    "neg_log_loss",
    "accuracy",
    "brier_score_loss",
    "roc_auc",
    "average_precision",
]
scores = run_cross_validation(complete_patient_timepoint_means, features_to_test, scoring_names)

# summary statistics
summary_scores = scores.groupby(level=0).describe()
summary_scores.loc[:, pd.IndexSlice[map(lambda x: "test_" + x, scoring_names), ["mean", "std", "50%"]]]
roc_scores, pr_scores = run_cross_validation_plots(complete_patient_timepoint_means, features_to_test)
roc_scores = roc_scores.fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
pr_scores = pr_scores.fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
roc_scores = roc_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))
pr_scores = pr_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))

calculate_significance(scores).round(4)

# how does DSP data compare to the predictive ability of ER and pam50 status (known predictors)
features_to_test = {
    "markers": (list(map(lambda x: x+"_mean", markers)), ["B", "R"]),
    "markers_ER_pam50": (list(map(lambda x: x+"_mean", markers + ["ER", "pam50"])), ["B", "R"]),
    "ER_pam50": (list(map(lambda x: x+"_mean", ["ER", "pam50"])), ["B", "R"]),
}
scores = run_cross_validation(complete_patient_timepoint_means, features_to_test, scoring_names)
summary_scores = scores.groupby(level=0).describe()
summary_scores.loc[:, pd.IndexSlice[map(lambda x: "test_" + x, scoring_names), ["mean", "std", "50%"]]]
roc_scores, pr_scores = run_cross_validation_plots(complete_patient_timepoint_means, features_to_test)
roc_scores = roc_scores.fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
pr_scores = pr_scores.fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
roc_scores = roc_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))
pr_scores = pr_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))
calculate_significance(scores).round(4)

# # How does DSP protein data compare to bulk rna data in terms of predictive ability?
bulk_input = pd.read_csv("bulk_input.txt", sep="\t").rename({"B2M": "Beta_2_microglobulin", "CD3E": "CD3", "CD8A": "CD8", "CTNNB1": "Beta_Catenin", "MKI67": "Ki_67", "CD276": "B7_H3", "PTPRC": "CD45", "BCL2": "Bcl_2", "AKT1": "AKT", "ITGAX": "CD11c", "CEACAM8": "CD66b", "C10orf54": "VISTA", "PDCD1": "PD1", "CD274": "PD_L1", "INDO": "IDO_1", "TMEM173": "STING_TMEM173", "LAG3": "Lag3", "ICOS": "ICOS_CD278", "TNFSF4": "OX40L_CD252_TXGP1", "VTCN1": "B7_H4_VTCN1"}, axis=1)

rna_protein_common_markers = list(set(bulk_input.columns).intersection(set(markers)))
rna_protein_common_markers
#########Bulk RNA processing
# Collapse multiple tissue samples for a patient timepoint into their mean and take B and R timepoints
bulk_rna_patient_timepoint_means = (
    bulk_input.loc[
        bulk_input.timepoint.isin(["B", "R"]),
        rna_protein_common_markers + ["patient", "timepoint"],
    ]
    .groupby(["patient", "timepoint"])
    .mean()
    .rename(lambda x: x + "_mean", axis=1)
)

# The patients with complete cases for B and R
complete_bulk_rna_patient_timepoint = bulk_rna_patient_timepoint_means.index.levels[0][
        bulk_rna_patient_timepoint_means.loc[pd.IndexSlice[:, ["B", "R"]], :]
        .reset_index("timepoint")
        .groupby("patient", sort=False)["timepoint"]
        .nunique()
        == 2
    ]

# Calculate RminusB
RminusB = bulk_rna_patient_timepoint_means.loc[pd.IndexSlice[:, "R"], :].droplevel(axis=0, level=1)  - bulk_rna_patient_timepoint_means.loc[pd.IndexSlice[:, "B"], :].droplevel(axis=0, level=1)
RminusB["timepoint"] = "RminusB"

# Add to dataframe
bulk_rna_patient_timepoint_means = pd.concat([bulk_rna_patient_timepoint_means.reset_index(level=1), RminusB], axis=0)
# Reshape to wide format
bulk_rna_patient_timepoint_means = bulk_rna_patient_timepoint_means.reset_index().pivot_table(
    index="patient", values=map(lambda x: x + "_mean", rna_protein_common_markers), columns="timepoint"
)

# Add on the response data
bulk_rna_patient_timepoint_means["pCR"] = raw_df.groupby("patient")["pCR"].first()

# Restrict to patients that are common between DSP protein and rna. Make sure patients in same order
common_complete_dsp_rna_patient_timepoint = list(
    set(complete_bulk_rna_patient_timepoint)
    .intersection(set(complete_patient_timepoint_means.index.get_level_values(0)))
)
common_complete_patient_timepoint_means = complete_patient_timepoint_means.reindex(
    common_complete_dsp_rna_patient_timepoint
)

common_bulk_rna_patient_timepoint_means = bulk_rna_patient_timepoint_means.reindex(
    common_complete_dsp_rna_patient_timepoint
)

# merge into single wide dataframe
common_modality_patient_timepoint_means = pd.concat([
    common_bulk_rna_patient_timepoint_means.drop("pCR", axis=1, level=0).rename(lambda x: "bulk_rna_"+x, axis=1, level=0),
    common_complete_patient_timepoint_means.drop("pCR", axis=1, level=0).rename(lambda x: "dsp_protein_"+x, axis=1, level=0),
], axis=1)
common_modality_patient_timepoint_means["pCR"] = common_complete_patient_timepoint_means.pCR

features_to_test = {
    "bulk_rna": (list(map(lambda x: "bulk_rna_"+x+"_mean", rna_protein_common_markers)), ["B", "R"]),
    "dsp_protein": (list(map(lambda x: "dsp_protein_"+x+"_mean", rna_protein_common_markers)), ["B", "R"]),
}

scores = run_cross_validation(common_modality_patient_timepoint_means, features_to_test, scoring_names)
summary_scores = scores.groupby(level=0).describe()
summary_scores.loc[:, pd.IndexSlice[map(lambda x: "test_" + x, scoring_names), ["mean", "std", "50%"]]]
roc_scores, pr_scores = run_cross_validation_plots(common_modality_patient_timepoint_means, features_to_test)
roc_scores = roc_scores.fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
pr_scores = pr_scores.fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
roc_scores = roc_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))
pr_scores = pr_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))
calculate_significance(scores).round(4)


# # How does having additional regions help with the prediction?
patient_timepoint_sem = raw_df.loc[raw_df.timepoint.isin(["B", "R"]), markers + igg + ["patient", "timepoint"]].groupby(["patient", "timepoint"]).apply(lambda x: x.std() / np.sqrt(x.count())).rename(lambda x: x+"_sem", axis=1)

# patients with more than 3 tissues at both timepoints
patient_with_more_3_tissues = (raw_df.loc[raw_df.timepoint.isin(["B", "R"]), ["patient", "timepoint"]].reset_index().groupby(["patient", "timepoint"]).count() >= 3).groupby("patient").all()
patient_with_more_3_tissues_and_both_timepoints = patient_with_more_3_tissues.index[patient_with_more_3_tissues.values[:,0]].intersection(patient_timepoint_sem.index.levels[0][
        patient_timepoint_sem.loc[pd.IndexSlice[:, ["B", "R"]], :]
        .reset_index("timepoint")
        .groupby("patient", sort=False)["timepoint"]
        .nunique()
        == 2
    ])
print(f"number of patients remaining {patient_with_more_3_tissues_and_both_timepoints.shape[0]}")
# convert to wide format and get complete patients only
complete_patient_timepoint_sem = patient_timepoint_sem.reset_index().pivot_table(
    index="patient", values=map(lambda x: x + "_sem", markers + igg), columns="timepoint"
).reindex(patient_with_more_3_tissues_and_both_timepoints)
# merge with mean datas
complete_patient_timepoint_means_and_sem = pd.concat([
    complete_patient_timepoint_means.reindex(patient_with_more_3_tissues_and_both_timepoints),
    complete_patient_timepoint_sem
], axis=1)

features_to_test = {
    "immune_means_and_tumor_sem": (list(map(lambda x: x + "_mean", immune_markers)) + list(map(lambda x: x + "_sem", tumor_markers)), ["B", "R"]),
    "immune_means_and_tumor_means": (list(map(lambda x: x + "_mean", immune_markers)) + list(map(lambda x: x + "_mean", tumor_markers)), ["B", "R"]),
    "immune_sem_and_tumor_sem": (list(map(lambda x: x + "_sem", immune_markers)) + list(map(lambda x: x + "_sem", tumor_markers)), ["B", "R"]),
    "immune_sem_and_tumor_means": (list(map(lambda x: x + "_sem", immune_markers)) + list(map(lambda x: x + "_means", tumor_markers)), ["B", "R"]),
}

scores = run_cross_validation(complete_patient_timepoint_means_and_sem, features_to_test, scoring_names)
# summary statistics
summary_scores = scores.groupby(level=0).describe()
summary_scores.loc[:, pd.IndexSlice[map(lambda x: "test_" + x, scoring_names), ["mean", "std", "50%"]]]

roc_scores, pr_scores = run_cross_validation_plots(complete_patient_timepoint_means_and_sem, features_to_test)
roc_scores = roc_scores.fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
pr_scores = pr_scores.fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
roc_scores = roc_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))
pr_scores = pr_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))
calculate_significance(scores).round(4)

# # Is full model better than null model (permuted labels)

score, permutation_scores, pvalue = do_permutation_test_vs_shuffled_null(complete_patient_timepoint_means, (list(map(lambda x: x+"_mean", markers)), ["B", "R"]))
n_classes = 2
plt.hist(permutation_scores, 20, label='Permutation scores',
         edgecolor='black')
ylim = plt.ylim()
plt.plot(2 * [score], ylim, '--g', linewidth=3,
         label='Classification Score'
         ' (p-value= %.4f)' % pvalue)

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')
plt.show()