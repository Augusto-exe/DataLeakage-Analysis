import signal
import sys

def signal_handler(sig, frame):
    print("Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

suffix = "_xgb_NN"
# Standard library imports
import random
from typing import List, Dict, Union

# Third-party library imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import xgboost as xgb

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    KFold,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    RFE,
    RFECV,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# External library imports
from pmlb import fetch_data

# Local imports
from utils import *

# Visualization imports
from matplotlib.axes._axes import Axes

print("Starting model script...")

#"LR",
target_algorithms = ["KNN",  "LR", "NB","DT", "RF", "SVC"] #

# New algorithms for additional testing
new_algorithms = ["KNN",  "LR", "NB","DT", "RF", "SVC","XGB", "NN"]

sub_ds_list = ["GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1","agaricus_lepiota","mushroom","ring","twonorm"]
partial_ds = ["waveform_40","waveform_21","movement_libras","satimage","chess","kr_vs_kp","optdigits","splice","texture","sonar","molecular_biology_promoters","mfeat_fourier","analcatdata_authorship","tokyo1","soybean","mfeat_karhunen"]
#ds_list = ["optdigits","splice","texture","sonar","molecular_biology_promoters","mfeat_fourier","analcatdata_authorship","tokyo1","soybean","mfeat_karhunen"]
#ds_list = ["GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1","agaricus_lepiota","mushroom","ring","twonorm","clean1","dna","phoneme","mfeat_pixel","banana","mfeat_factors","spambase","Hill_Valley_with_noise","Hill_Valley_without_noise","waveform_40","waveform_21","movement_libras","satimage","chess","kr_vs_kp","optdigits","splice","texture","sonar","molecular_biology_promoters","mfeat_fourier","analcatdata_authorship","tokyo1","soybean","mfeat_karhunen"]
additional_15_ds =  ["mnist", "poker", "kddcup","clean2", "coil2000", "connect_4", "sleep", "fars", "adult", "spectf", "shuttle", "magic", "letter", "krkopt", "dis"]
ds_list =[ "fars", "adult", "spectf", "shuttle", "magic", "letter", "krkopt", "dis"] #["poker"]
#ds_list =["kddcup"] #"clean2", "coil2000", "connect_4", "sleep", "fars", "adult", "spectf", "shuttle", "magic", "letter", "krkopt", "dis","mnist", "poker", 
#additional_15_ds
#["mnist", "poker", "kddcup",
new_list = ["satimage","chess","kr_vs_kp","optdigits","splice","texture","sonar","molecular_biology_promoters","mfeat_fourier","analcatdata_authorship","tokyo1","soybean","mfeat_karhunen"]
#ds_list = new_list
NUM_TRIALS = 6

class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for XGBClassifier that handles label encoding automatically"""
    def __init__(self, **kwargs):
        self.xgb_params = kwargs
        self.model = xgb.XGBClassifier(**kwargs)
        self.label_encoder = LabelEncoder()
        self._fitted = False
        
    def fit(self, X, y, **fit_params):
        # Encode labels to start from 0
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded, **fit_params)
        self._fitted = True
        return self
        
    def predict(self, X):
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        # Get predictions and decode back to original labels
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
        
    def predict_proba(self, X):
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        if deep:
            return self.xgb_params.copy()
        else:
            return self.xgb_params
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        self.xgb_params.update(params)
        self.model = xgb.XGBClassifier(**self.xgb_params)
        return self
        
    def __sklearn_is_fitted__(self):
        """Check if the estimator is fitted."""
        return self._fitted
        
    def __getattr__(self, name):
        # Delegate other attributes to the underlying model
        return getattr(self.model, name)

def get_estimator(alg):
  if alg == "KNN":
    return KNeighborsClassifier()
  elif alg == "SVC":
    return SVC(kernel="linear", C=1, random_state = 42,max_iter=50000) #500000
  elif alg == "LR":
    return LogisticRegression(random_state=42,max_iter=5000,solver="sag")
  elif alg == "NB":
    return GaussianNB()
  elif alg == "DT":
    return DecisionTreeClassifier(random_state=42)
  elif alg == "NN":
    return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
  elif alg == "XGB":
    return XGBClassifierWrapper(random_state=42, eval_metric='logloss', verbosity=0)
  else:
    return RandomForestClassifier(random_state=42)

def get_scoring(n_classes):
  if n_classes > 2:
    scoring = ['accuracy', 'balanced_accuracy','f1_macro','precision_macro','recall_macro']
  else:
    scoring = ['accuracy', 'balanced_accuracy','f1','precision','recall']
  return scoring

def multiple_score(n_classes, y_real, y_pred):
    if n_classes > 2:
        score = {
            'test_accuracy': accuracy_score(y_real, y_pred),
            'test_balanced_accuracy': balanced_accuracy_score(y_real, y_pred),
            'test_f1': f1_score(y_real, y_pred, average='macro', zero_division=0),
            'test_precision': precision_score(y_real, y_pred, average='macro', zero_division=0),
            'test_recall': recall_score(y_real, y_pred, average='macro', zero_division=0)
        }
    else:
        score = {
            'test_accuracy': accuracy_score(y_real, y_pred),
            'test_balanced_accuracy': balanced_accuracy_score(y_real, y_pred),
            'test_f1': f1_score(y_real, y_pred, zero_division=0),
            'test_precision': precision_score(y_real, y_pred, zero_division=0),
            'test_recall': recall_score(y_real, y_pred, zero_division=0)
        }
    return score

def make_dict(n_classes, results):
    if n_classes > 2:
        score = {
            'test_accuracy': results["mean_test_accuracy"][np.nonzero(results["rank_test_accuracy"] == 1)[0][0]],
            'test_balanced_accuracy': results["mean_test_balanced_accuracy"][np.nonzero(results["rank_test_balanced_accuracy"] == 1)[0][0]],
            'test_f1': results["mean_test_f1_macro"][np.nonzero(results["rank_test_f1_macro"] == 1)[0][0]],
            'test_precision': results["mean_test_precision_macro"][np.nonzero(results["rank_test_precision_macro"] == 1)[0][0]],
            'test_recall': results["mean_test_recall_macro"][np.nonzero(results["rank_test_recall_macro"] == 1)[0][0]]
        }
    else:
        score = {
            'test_accuracy': results["mean_test_accuracy"][np.nonzero(results["rank_test_accuracy"] == 1)[0][0]],
            'test_balanced_accuracy': results["mean_test_balanced_accuracy"][np.nonzero(results["rank_test_balanced_accuracy"] == 1)[0][0]],
            'test_f1': results["mean_test_f1"][np.nonzero(results["rank_test_accuracy"] == 1)[0][0]],
            'test_precision': results["mean_test_precision"][np.nonzero(results["rank_test_precision"] == 1)[0][0]],
            'test_recall': results["mean_test_recall"][np.nonzero(results["rank_test_recall"] == 1)[0][0]]
        }
    return score

def make_dict_mean(n_classes, results):
    if n_classes > 2:
        score = {
            'test_accuracy': results["test_accuracy"].mean(),
            'test_balanced_accuracy': results["test_balanced_accuracy"].mean(),
            'test_f1': results["test_f1_macro"].mean(),
            'test_precision': results["test_precision_macro"].mean(),
            'test_recall': results["test_recall_macro"].mean()
        }
    else:
        score = {
            'test_accuracy': results["test_accuracy"].mean(),
            'test_balanced_accuracy': results["test_balanced_accuracy"].mean(),
            'test_f1': results["test_f1"].mean(),
            'test_precision': results["test_precision"].mean(),
            'test_recall': results["test_recall"].mean()
        }
    return score

def produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str,
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str,
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.

    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)

    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape) < p_miss).double()

    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan

    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}

def get_imputer(imp):
  if imp == 'mean':
    return SimpleImputer(strategy='mean')
  elif imp == 'median':
    return SimpleImputer(strategy='median')
  elif imp == 'KNN':
    return KNNImputer(n_neighbors=5)
  elif imp == 'iterative':
    return IterativeImputer(random_state=42)
  else:
    return SimpleImputer(strategy='mean')

def insert_missing_in_df(df,perc):
  tgt = df['target'].to_numpy()
  columns = df.columns
  X_miss = produce_NA(df.drop('target', axis=1).to_numpy(),perc,"MAR",p_obs=0.9)
  df2 = pd.DataFrame(X_miss['X_incomp'].cpu().detach().numpy(), columns = columns[:-1])
  df2['target'] = tgt
  return df2