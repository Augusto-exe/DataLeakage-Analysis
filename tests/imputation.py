import sys
sys.path.append('..')  # Adjust the path as needed
from model_utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_value_imputation_test():
    print("Value Imputation")
    target_ds_list = ds_list
    print("Datasets to be used: ", target_ds_list)

    df_impute = pd.DataFrame()
    perc_miss = [0.05,0.1,0.2,0.3]
    imp_list = ['mean','median','KNN']#,'iterative']
    for ds in target_ds_list:
      print("\n TESTING ds: _", ds, "_\n")
      for p in perc_miss:
        print("p miss: ",p)
        cv = StratifiedKFold(n_splits=5,shuffle = True, random_state=42 )
        df = insert_missing_in_df(fetch_data(ds),p)
        scoring = get_scoring(df['target'].nunique())
        target_algorithms = new_algorithms
        for alg in target_algorithms:
          print("alg: ",alg)

          estimator = get_estimator(alg)
          for imp in imp_list:
            imputer = get_imputer(imp)
            # Arrays to store scores
            cv_scores_leak = np.zeros(NUM_TRIALS)
            cv_scores_leak_all = []
            cv_scores_noleak  = np.zeros(NUM_TRIALS)
            cv_scores_noleak_all  = []
            # Loop for each trial
            for i in range(NUM_TRIALS):

              train_data = df.drop('target', axis=1)
              train_target = df['target']
              # Create the pipeline with linear classifier step
              pipeline = Pipeline([
                  ('imputer', imputer),
                  ('scaler', StandardScaler()),
                  ('classifier', estimator)
              ])
              cv = StratifiedKFold(n_splits=4,shuffle = True, random_state=i)

              # Evaluate the pipeline using cross-validation on the training data
              cv_scores_noleak_all.append(make_dict_mean(df['target'].nunique(),cross_validate(pipeline, train_data, train_target, cv=cv,scoring=scoring,return_train_score=False)))


              # imput using all data

              X = train_data.copy()
              impt_2 = imputer.fit(train_data)
              X = impt_2.transform(X)
              train_X = pd.DataFrame(X,columns = train_data.columns)

              # Normalize the entire data
              reg = Pipeline([
                  ('scaler', StandardScaler()),
                  ('classifier', estimator)
              ])

              # Evaluate the pipeline using cross-validation on the entire data
              cv_scores_leak_all.append(make_dict_mean(df['target'].nunique(),cross_validate(reg, train_X, train_target, cv=cv,scoring=scoring,return_train_score=False)))

              cv_scores_noleak[i] = cv_scores_noleak_all[i]["test_accuracy"]
              cv_scores_leak[i] = cv_scores_leak_all[i]["test_accuracy"]

              cv_scores_leak_all[i]["ds"] = ds
              cv_scores_leak_all[i]["iteration"] = i
              cv_scores_leak_all[i]["alg"] = alg
              cv_scores_leak_all[i]["leak"] = True
              cv_scores_leak_all[i]["imputer"] = imp
              cv_scores_leak_all[i]["perc-miss"] = p

              cv_scores_noleak_all[i]["ds"] = ds
              cv_scores_noleak_all[i]["iteration"] = i
              cv_scores_noleak_all[i]["alg"] = alg
              cv_scores_noleak_all[i]["leak"] = False
              cv_scores_noleak_all[i]["imputer"] = imp
              cv_scores_noleak_all[i]["perc-miss"] = p

              df_temp = pd.DataFrame([cv_scores_leak_all[i],cv_scores_noleak_all[i]])
              df_impute  = pd.concat([df_impute ,df_temp], ignore_index=True)

              df_impute.to_csv("imputation_results"+suffix+".csv", index=False)