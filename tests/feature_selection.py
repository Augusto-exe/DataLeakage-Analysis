import sys
sys.path.append('..')  # Adjust the path as needed
from model_utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectPercentile


def run_feature_selection_test():
    print("Feature Selection")
    target_ds_list = ds_list
    print("Datasets to be used: ", target_ds_list)

    ## Version 2
    df_feat = pd.DataFrame()
    for ds in target_ds_list:
      print("\n TESTING ds: _", ds, "_\n")
      cv = StratifiedKFold(n_splits=5,shuffle = True, random_state=42 )
      df = fetch_data(ds)

      percentile_list = [1,5,10,20]
      scoring = get_scoring(df['target'].nunique())
      target_algorithms = new_algorithms
      for alg in target_algorithms:
        print("alg: ",alg)
        for perc in percentile_list:
          print("perc: ",perc)
          estimator = get_estimator(alg)

          # Arrays to store scores
          cv_scores_leak = np.zeros(NUM_TRIALS)
          cv_scores_leak_all = []
          cv_scores_noleak  = np.zeros(NUM_TRIALS)
          cv_scores_noleak_all  = []

          X = df.drop('target', axis=1)
          y = df['target']
          actual_perc = perc
          if(len(X.columns)*(perc/100) < 1):
            print("Skipping feature selection for ", ds, " with perc ", perc, " as it results in less than 1 feature.")
            actual_perc = 100/ len(X.columns)
            print("Using actual perc ", actual_perc)
          leak_feature_selector = SelectPercentile(percentile = actual_perc)
          X_leak = leak_feature_selector.fit_transform(X, y)
          if len(X_leak[0]) > 0:
            # Loop for each trial
            for i in range(NUM_TRIALS):
                train_data, test_data, train_target, test_target = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=i)


                no_leak_feature_selector = SelectPercentile(percentile = actual_perc)
                train_no_leak = no_leak_feature_selector.fit_transform(train_data,train_target)
                no_leak_model = estimator.fit(train_no_leak, train_target)
                test_no_leak = no_leak_feature_selector.transform(test_data)
                cv_scores_noleak_all.append(multiple_score(df['target'].nunique(),test_target,no_leak_model.predict(test_no_leak)))


                train_leak = leak_feature_selector.transform(train_data)
                leak_model = estimator.fit(train_leak, train_target)
                test_leak = leak_feature_selector.transform(test_data)
                cv_scores_leak_all.append(multiple_score(df['target'].nunique(),test_target,leak_model.predict(test_leak)))

                cv_scores_noleak[i] = cv_scores_noleak_all[i]["test_accuracy"]
                cv_scores_leak[i] = cv_scores_leak_all[i]["test_accuracy"]

                cv_scores_leak_all[i]["ds"] = ds
                cv_scores_leak_all[i]["iteration"] = i
                cv_scores_leak_all[i]["alg"] = alg
                cv_scores_leak_all[i]["leak"] = True
                cv_scores_leak_all[i]["perc"] = perc

                cv_scores_noleak_all[i]["ds"] = ds
                cv_scores_noleak_all[i]["iteration"] = i
                cv_scores_noleak_all[i]["alg"] = alg
                cv_scores_noleak_all[i]["leak"] = False
                cv_scores_noleak_all[i]["perc"] = perc

                df_temp = pd.DataFrame([cv_scores_leak_all[i],cv_scores_noleak_all[i]])
                df_feat = pd.concat([df_feat,df_temp], ignore_index=True)
                print("CV scores (with data leakage) ",alg, " in ", ds, ": ", cv_scores_leak[i])
                print("CV scores (without data leakage) ",alg, " in ", ds, ": ", cv_scores_noleak[i])
                df_feat.to_csv("feature_selection_results"+suffix+".csv", index=False)
