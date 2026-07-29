import sys
sys.path.append('..')  # Adjust the path as needed
from model_utils import *
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from model_utils import get_estimator, get_scoring, make_dict, make_dict_mean
from pmlb import fetch_data
import gc


def run_hyperparameter_tuning_test():
    hyperparameterDict = { "KNN": {
                            'n_neighbors': [3, 5, 7, 9, 11],
                            'metric': ['euclidean', 'manhattan']},
                        "SVC": {
                            'C': [0.01, 0.1, 1, 10, 100],
                            'kernel': ['linear', 'rbf', 'poly'],},
                        "LR": {
                            'solver': [ 'saga', 'lbfgs'],
                            'C': [0.01, 0.1, 1, 10, 100],},
                        "NB": {
                            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
                        "DT": {
                            'max_depth': [3, 5, 8, 10, None],},
                        "RF": {
                            'n_estimators': [100, 200, 500],
                            'max_depth': [3, 5, 8, 10, None],},
                        "NN": {  
                            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                            'activation': ['relu', 'tanh'],
                            'alpha': [0.0001, 0.001, 0.01],
                            'max_iter': [500, 1000]
                          },
                        "XGB": {
                            'n_estimators': [100, 200, 300],
                            'max_depth': [3, 5, 7],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'subsample': [0.8, 1.0]
                          }
    }
    df_hyp = pd.DataFrame()

    target_columns_clf = []
    target_columns = []

    target_ds_list = ["clean2"]#ds_list
    print("Datasets to be used: ", target_ds_list)

    print("HiperParameter")
    for ds in target_ds_list:
      print("\n TESTING ds: _", ds, "_\n")
      cv = StratifiedKFold(n_splits=5,shuffle = True, random_state=42 )
      df = fetch_data(ds)
      scoring = get_scoring(df['target'].nunique())
      target_algorithms = new_algorithms
      for alg in ["NN"]:#target_algorithms:
        print("alg: ",alg)
        estimator = get_estimator(alg)

        cv_scores_leak = np.zeros(NUM_TRIALS)
        cv_scores_leak_all = []
        cv_scores_noleak  = np.zeros(NUM_TRIALS)
        cv_scores_noleak_all  = []

        p_grid = hyperparameterDict[alg]

        for i in range(NUM_TRIALS):
            print("\n Trial: ", i+1, " of ", NUM_TRIALS,"\n")
            train_data = df.drop('target', axis=1)
            train_target = df['target']
            inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)

            clf = GridSearchCV(estimator=estimator, param_grid=p_grid, cv=outer_cv,scoring=scoring,refit='accuracy',n_jobs=1)
            clf.fit(train_data,train_target)
            clf_results = make_dict(df['target'].nunique(),clf.cv_results_)
            cv_scores_leak_all.append(clf_results)

            clf = GridSearchCV(estimator=estimator, param_grid=p_grid, cv=inner_cv,refit='accuracy',n_jobs=1)
            nested_score = cross_validate(clf, X=train_data, y=train_target, cv=outer_cv,scoring=scoring)
            cv_scores_noleak_all.append(make_dict_mean(df['target'].nunique(),nested_score))

            cv_scores_noleak[i] = cv_scores_noleak_all[i]["test_accuracy"]
            cv_scores_leak[i] = cv_scores_leak_all[i]["test_accuracy"]

            cv_scores_leak_all[i]["ds"] = ds
            cv_scores_leak_all[i]["iteration"] = i
            cv_scores_leak_all[i]["alg"] = alg
            cv_scores_leak_all[i]["leak"] = True

            cv_scores_noleak_all[i]["ds"] = ds
            cv_scores_noleak_all[i]["iteration"] = i
            cv_scores_noleak_all[i]["alg"] = alg
            cv_scores_noleak_all[i]["leak"] = False

            df_temp = pd.DataFrame([cv_scores_leak_all[i],cv_scores_noleak_all[i]])
            df_hyp = pd.concat([df_hyp,df_temp], ignore_index=True)
            print("CV scores (with data leakage) ",alg, " in ", ds, ": ", cv_scores_leak[i])
            print("CV scores (without data leakage) ",alg, " in ", ds, ": ", cv_scores_noleak[i])
            df_hyp.to_csv("hyperparameter_tuning_results"+suffix+"CLEAN2NN.csv", index=False)
            gc.collect()


