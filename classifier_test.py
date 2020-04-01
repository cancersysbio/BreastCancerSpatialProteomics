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
from itertools import combinations
%reload_ext blackcellmagic
from joblib import Parallel, delayed
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
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

random_state = 100

# input training and testing data files with correct name and path
nanostring_dat_train = 
nanostring_dat_test =

raw_train = pd.read_csv(nanostring_dat_train, sep="\t", index_col=0)
raw_train = raw_train.rename(mapper=lambda x: x.replace(".", "_"), axis=1)
raw_train["pCR"] = 1 - raw_train["no_pCR"]
raw_test = pd.read_csv(nanostring_dat_test, sep="\t", index_col=0)
raw_test = raw_test.rename(mapper=lambda x: x.replace(".", "_"), axis=1)

markers = [
    "Beta.Catenin",
    "pS6",
    "PTEN",
    "P.ERK",
    "S6",
    "Ki.67",
    "Beta.2.microglobulin",
    "AKT",
    "p.AKT",
    "Her2",
    "Pan.Cytokeratin",
    "CD8",
    "B7.H3",
    "CD4",
    "CD68",
    "GZMB",
    "CD3",
    "CD66B",
    "VISTA",
    "CD44",
    "PD.L1",
    "CD45RO",
    "Bcl.2",
    "B7.H4.VTCN1",
    "STING.TMEM173",
    "IDO.1",
    "CD11c"
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
    "p.AKT",
    "Her2",
    "Pan.Cytokeratin",
]
tumor_markers = list(map(lambda x: x.replace(".", "_"), tumor_markers))
immune_markers = [
    "CD8",
    "CD4",
    "CD68",
    "GZMB",
    "CD3",
    "CD66B",
    "VISTA",
    "CD44"
    "CD45RO",
    "STING.TMEM173",
    "CD11c",
    "B7.H3",
    "PD.L1",
    "B7.H4.VTCN1",
    "IDO.1"
]
immune_markers = list(map(lambda x: x.replace(".", "_"), immune_markers))

raw_train[markers] = StandardScaler().fit_transform(raw_train[markers])
raw_test[markers] = StandardScaler().fit_transform(raw_test[markers])

# Collapse multiple tissue samples for a patient timepoint into their mean and take B and R timepoints
train_means = raw_train.loc[raw_train.timepoint.isin(["B", "R"]), markers + ["patient", "timepoint"]].groupby(["patient", "timepoint"]).mean().rename(lambda x: x+"_mean", axis=1)
test_means = raw_test.loc[raw_test.timepoint.isin(["B", "R"]), markers + ["patient", "timepoint"]].groupby(["patient", "timepoint"]).mean().rename(lambda x: x+"_mean", axis=1)

# Get complete cases only (must have both B and R timepoints)
complete_test_means = test_means.loc[
    test_means.index.levels[0][
        test_means.loc[pd.IndexSlice[:, ["B", "R"]], :]
        .reset_index("timepoint")
        .groupby("patient", sort=False)["timepoint"]
        .nunique()
        == 2
    ]
]

complete_train_means = train_means.loc[
    train_means.index.levels[0][
        train_means.loc[pd.IndexSlice[:, ["B", "R"]], :]
        .reset_index("timepoint")
        .groupby("patient", sort=False)["timepoint"]
        .nunique()
        == 2
    ]
]

# Reshape to wide format
complete_train_means = complete_train_means.reset_index().pivot_table(
    index="patient", values=map(lambda x: x + "_mean", markers), columns="timepoint"
)
complete_test_means = complete_test_means.reset_index().pivot_table(
    index="patient", values=map(lambda x: x + "_mean", markers), columns="timepoint"
)

# Add on the response data
complete_test_means["pCR"] = raw_test.groupby("patient")["pCR"].first()
complete_train_means["pCR"] = raw_train.groupby("patient")["pCR"].first()

# add in ER and pam50 Status
ERstatus = raw_train.groupby("patient")[["ER"]].first()
pam50status = raw_train.groupby("patient")["pam50"].first().fillna("other") # pam50 is unknown for one patient, fill it with "other"
complete_train_means["ER_mean", "B"] = ERstatus # pardon the mean naming, makes code simpler
complete_train_means["pam50_mean", "B"] = pam50status.astype("category").cat.codes # pardon the mean naming, makes code simpler

ERstatus = raw_test.groupby("patient")[["ER"]].first()
pam50status = raw_test.groupby("patient")["pam50"].first().fillna("other") # pam50 is unknown for multiple patient, fill it with "other"
complete_test_means["ER_mean", "B"] = ERstatus # pardon the mean naming, makes code simpler
complete_test_means["pam50_mean", "B"] = pam50status.astype("category").cat.codes # pardon the mean naming, makes code simpler

# # Should not have any null values
assert complete_test_means.isnull().any().any() == False
assert complete_train_means.isnull().any().any() == False

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
    ## We'll do 100x repeated max-fold stratified cross-validation, then unpaired t-testing for significance
    scores = {}

    for feature, pipeline in pipelines_to_compare.items():
        scores[feature] = pd.DataFrame.from_records(
            cross_validate(
                pipeline,
                dataframe,
                dataframe.pCR,
                cv=RepeatedStratifiedKFold(random_state=random_state,
                    n_splits=dataframe.pCR.value_counts().min()//2, # max-folds for stratified is the minimum of number of samples in class
                    n_repeats=100 # TODO: since small dataset, probably should check what the max possible combinations are
                ),
                scoring=scoring_names,
                n_jobs=36,
            )
        )
    scores = pd.concat(scores.values(), keys=scores.keys(), names=["marker_group", "fold"])
    
    return scores,pipelines_to_compare

def make_roc_test(dataframe, dataframe_test, pipeline, name):
    """
    train the winning model using the train data and generate roc curve using the train data
    """
    pipeline.fit(dataframe, dataframe.pCR)
    pcr_pred = pipeline.decision_function(dataframe_test)
    score=roc_auc_score(dataframe_test.pCR, pcr_pred)
    fpr, tpr, thresholds = roc_curve(dataframe_test.pCR, pcr_pred)
    plt.plot(fpr, tpr, marker='.', label=name + " AUC= " + str(round(score,3)))
    # axis labels
    plt.xlabel('FPR (1-Specificity)')
    plt.ylabel('TRP (Sensitivity)')
    # show the legend
    plt.legend()
    plt.title('ROC')
    # save the plot
    plt.savefig((name+".pdf"), transparent=True)

def get_model_weights(dataframe, model_pipeline, features_to_use):
    """
    Train a model and then return model weights
    :param pd.DataFrame dataframe:  a wide dataframe, with single patient per row, multiindex columns w/ level 0 marker, level 1 timepoint. also must contain "pCR" as a column
    """
    model_pipeline.fit(dataframe, dataframe.pCR)
    
    weights = pd.DataFrame(model_pipeline[-1].coef_.T, index=dataframe.loc[:, pd.IndexSlice[features_to_use[0], features_to_use[1]]].columns, columns=["weight"])
    weights = weights.append(pd.DataFrame([[model_pipeline[-1].intercept_[0]]], index=pd.MultiIndex.from_tuples([("intercept", "NA")]), columns=["weight"])).rename_axis(["marker", "timepoint"])
    return weights

scoring_names = [
    "neg_log_loss",
    "accuracy",
    "brier_score_loss",
    "roc_auc",
    "average_precision",
]

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
    ## We'll do 100x repeated max-fold stratified cross-validation, then unpaired t-testing for significance
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
    return roc_score

# example commands for comparing timepoints, change features to test to compare other parameters
features_to_test = {
    "B": (list(map(lambda x: x+"_mean", markers)), "B"),
    "R": (list(map(lambda x: x+"_mean", markers)), "R"),
    "BandR": (list(map(lambda x: x+"_mean", markers)), ["B", "R"])
}
# other examples of features to test
features_to_test = {
    "markers": (list(map(lambda x: x+"_mean", markers)), ["B", "R"]),
    "tumor_markers": (list(map(lambda x: x+"_mean", tumor_markers)), ["B", "R"]),
    "immune_markers": (list(map(lambda x: x+"_mean", immune_markers)), ["B", "R"]),
}
features_to_test = {
    "markers": (list(map(lambda x: x+"_mean", markers)), ["B", "R"]),
    "markers_ER_pam50": (list(map(lambda x: x+"_mean", markers + ["ER", "pam50"])), ["B", "R"]),
    "ER_pam50": (list(map(lambda x: x+"_mean", ["ER", "pam50"])), ["B", "R"]),
}
scores,pipelines = run_cross_validation(complete_train_means, features_to_test, scoring_names)


roc_scores = run_cross_validation_plots(complete_train_means, features_to_test)
roc_scores = roc_scores.fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
roc_scores = roc_scores.groupby(["feature", "metric"]).apply(lambda x: pd.DataFrame([x.mean(axis=0), x.quantile(0.40, axis=0), x.quantile(0.66, axis=0)], index=["mean", "lower", "upper"]))
scores,pipelines = run_cross_validation(complete_train_means, features_to_test, scoring_names)

# fill in with appropriate AUCs
f, ax = plt.subplots()
ax.step(roc_scores.loc[pd.IndexSlice["B", "fpr", "mean"], :],
        roc_scores.loc[pd.IndexSlice["B", "tpr", "mean"], :],
        label="Pre-treatment (Mean AUC=0.628)",)
ax.step(roc_scores.loc[pd.IndexSlice["R", "fpr", "mean"], :],
        roc_scores.loc[pd.IndexSlice["R", "tpr", "mean"], :],
        label="On-treatment (Mean AUC=0.737)",)
ax.step(roc_scores.loc[pd.IndexSlice["BandR", "fpr", "mean"], :],
        roc_scores.loc[pd.IndexSlice["BandR", "tpr", "mean"], :],
        label="On- + Pre-treatment (Mean AUC=0.721)",)
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random Chance", alpha=0.8)
ax.legend(bbox_to_anchor=(.27, .33))
ax.set_xlim([0, 1])
ax.set_xlabel("FPR (1-Specificity)")
ax.set_ylim([0, 1])
ax.set_ylabel("TRP (Sensitivity)")
f.suptitle("ROC")
plt.savefig("timepoint_train_roc.pdf", transparent=True)

make_roc_test(complete_train_means, complete_test_means, pipelines["BandR"], "Pre-treatment and On-treatment")
best_model_weights=get_model_weights(complete_train_means,pipelines["BandR"], (list(map(lambda x: x+"_mean", markers)), ["B", "R"]))
best_model_weights.plot(kind="bar", figsize=(20,5))
make_roc_test(complete_train_means, complete_test_means, pipelines["B"], "Pre-treatment") # for other timepoints
