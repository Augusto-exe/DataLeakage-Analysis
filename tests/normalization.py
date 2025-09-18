import sys
sys.path.append('..')  # Adjust the path as needed
from model_utils import *
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from model_utils import get_estimator, get_scoring, make_dict_mean
from pmlb import fetch_data

def run_normalization_test():
    print("Normalization")
    target_ds_list = additional_15_ds
    print("Datasets to be used: ", target_ds_list)

    df_norm = pd.DataFrame()
    for ds in target_ds_list:
      print("\n TESTING ds: _", ds, "_\n")

      df = fetch_data(ds)

      for alg in target_algorithms:
        print("alg: ",alg)
        estimator = get_estimator(alg)
        # Split the data into training and test sets


        # Arrays to store scores
        cv_scores_leak = np.zeros(NUM_TRIALS)
        cv_scores_leak_all = []
        cv_scores_noleak  = np.zeros(NUM_TRIALS)
        cv_scores_noleak_all  = []
        scoring = get_scoring(df['target'].nunique())
        # Loop for each trial
        for i in range(NUM_TRIALS):
          #train_data, test_data, train_target, test_target = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=i)
          train_data = df.drop('target', axis=1)
          train_target = df['target']
          # Create the pipeline with linear classifier step
          pipeline = Pipeline([
              ('scaler', StandardScaler()),
              ('classifier', estimator)
          ])
          cv = StratifiedKFold(n_splits=4,shuffle = True, random_state=i)

          # Evaluate the pipeline using cross-validation on the training data
          cv_scores_noleak_all.append(make_dict_mean(df['target'].nunique(),cross_validate(pipeline, train_data, train_target, cv=cv,scoring=scoring,return_train_score=False)))
          cv_scores_noleak_all[i]["iteration"] = i
          cv_scores_noleak_all[i]["alg"] = alg
          cv_scores_noleak_all[i]["leak"] = False
          cv_scores_noleak_all[i]["ds"] = ds


          scaler = StandardScaler()
          X = train_data.copy()
          scaler = scaler.fit(train_data)
          X = scaler.transform(X)
          train_X = pd.DataFrame(X,columns = train_data.columns)

          # Normalize the entire data
          reg = Pipeline([
              ('classifier', estimator)
          ])

          cv_scores_leak_all.append(make_dict_mean(df['target'].nunique(),cross_validate(reg, train_X, train_target, cv=cv,scoring=scoring,return_train_score=False)))
          cv_scores_leak_all[i]["iteration"] = i
          cv_scores_leak_all[i]["alg"] = alg
          cv_scores_leak_all[i]["leak"] = True
          cv_scores_leak_all[i]["ds"] = ds

          cv_scores_noleak[i] = cv_scores_noleak_all[i]["test_accuracy"]
          cv_scores_leak[i] = cv_scores_leak_all[i]["test_accuracy"]

          df_temp = pd.DataFrame([cv_scores_leak_all[i],cv_scores_noleak_all[i]])
          df_norm = pd.concat([df_norm,df_temp], ignore_index=True)

          df_norm.to_csv("normalization_results.csv", index=False)
