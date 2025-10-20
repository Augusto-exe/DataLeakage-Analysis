import sys
sys.path.append('..')  # Adjust the path as needed
from model_utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def get_simplified_param_grid(alg):
    """Get simplified parameter grids for hyperparameter tuning"""
    param_grids =  {    "KNN": {
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
    return param_grids.get(alg, {})


def run_worst_case_leakage_test():
    """
    Test the worst case scenario where ALL data leakage happens:
    1. Normalization applied to entire dataset
    2. Missing value imputation on entire dataset  
    3. Feature selection on entire dataset
    4. SMOTE applied to entire dataset
    5. Hyperparameter tuning on entire dataset
    
    Compare against doing ALL preprocessing correctly within CV folds.
    """
    print("WORST CASE DATA LEAKAGE TEST")
    print("Testing scenario where ALL preprocessing steps leak data vs doing everything correctly")
    target_ds_list = ds_list
    print("Datasets to be used: ", target_ds_list)

    df_worst_case = pd.DataFrame()
    
    # Fixed parameters for consistent comparison
    MISSING_PERCENTAGE = 0.2  # 20% missing values
    FEATURE_SELECTION_PERCENTAGE = 10  # Select top 10% features
    
    for ds in target_ds_list:
        print(f"\n TESTING ds: _ {ds} _\n")
        
        try:
            df_original = fetch_data(ds)
            
            # Skip if dataset has only one class
            original_class_counts = df_original['target'].value_counts()
            if len(original_class_counts) < 2:
                print(f"Skipping {ds}: Only one class present")
                continue
            
            # Introduce missing values
            df_with_missing = insert_missing_in_df(df_original, MISSING_PERCENTAGE)
            
            print(f"Dataset: {ds}")
            print(f"Original shape: {df_original.shape}")
            print(f"Class distribution: {dict(original_class_counts)}")
            print(f"Missing values introduced: {MISSING_PERCENTAGE*100}%")
            print(f"Feature selection: top {FEATURE_SELECTION_PERCENTAGE}%")
            
            scoring = get_scoring(df_with_missing['target'].nunique())
            target_algorithms = new_algorithms
            
            for alg in target_algorithms:
                print(f"\nalg: {alg}")
                
                # Arrays to store scores
                cv_scores_leak = np.zeros(NUM_TRIALS)
                cv_scores_leak_all = []
                cv_scores_noleak = np.zeros(NUM_TRIALS)
                cv_scores_noleak_all = []

                # Loop for each trial
                for i in range(NUM_TRIALS):
                    print(f"  Trial {i+1}/{NUM_TRIALS}")
                    
                    train_data = df_with_missing.drop('target', axis=1)
                    train_target = df_with_missing['target']
                    
                    # Get fresh estimator for each trial
                    estimator_correct = get_estimator(alg)
                    estimator_leakage = get_estimator(alg)
                    
                    # Calculate parameters for SMOTE
                    class_counts = df_with_missing['target'].value_counts()
                    min_class_size = min(class_counts)
                    k_neighbors = min(5, max(1, min_class_size - 1))
                    
                    # =================================================================
                    # CORRECT APPROACH: All preprocessing within CV folds
                    # =================================================================
                    
                    pipeline_correct = ImbPipeline([
                        ('imputer', KNNImputer(n_neighbors=5)),
                        ('scaler', StandardScaler()),
                        ('feature_selector', SelectPercentile(percentile=FEATURE_SELECTION_PERCENTAGE)),
                        ('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)),
                        ('classifier', estimator_correct)
                    ])
                    
                    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
                    
                    # Get parameter grid for hyperparameter tuning
                    param_grid = get_simplified_param_grid(alg)
                    
                    if param_grid:
                        # Correct approach: Hyperparameter tuning with nested CV
                        grid_search_correct = GridSearchCV(
                            pipeline_correct, param_grid, 
                            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=i),
                            scoring='accuracy', refit=True
                        )
                        
                        # Nested CV: Grid search happens within each outer CV fold
                        cv_scores_noleak_all.append(
                            make_dict_mean(
                                df_with_missing['target'].nunique(),
                                cross_validate(grid_search_correct, train_data, train_target, 
                                             cv=cv, scoring=scoring, return_train_score=False)
                            )
                        )
                    else:
                        # No hyperparameter tuning for this algorithm
                        cv_scores_noleak_all.append(
                            make_dict_mean(
                                df_with_missing['target'].nunique(),
                                cross_validate(pipeline_correct, train_data, train_target, 
                                             cv=cv, scoring=scoring, return_train_score=False)
                            )
                        )
                    
                    # =================================================================
                    # WORST CASE LEAKAGE: All preprocessing on entire dataset first
                    # =================================================================
                    
                    # Step 1: Impute missing values using entire dataset (LEAKAGE)
                    imputer = KNNImputer(n_neighbors=5)
                    X_imputed = imputer.fit_transform(train_data)
                    X_imputed_df = pd.DataFrame(X_imputed, columns=train_data.columns)
                    
                    # Step 2: Normalize using entire dataset (LEAKAGE)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_imputed_df)
                    X_scaled_df = pd.DataFrame(X_scaled, columns=train_data.columns)
                    
                    # Step 3: Feature selection using entire dataset (LEAKAGE)
                    # Check if we have enough features for the percentage
                    n_features = len(X_scaled_df.columns)
                    actual_perc = FEATURE_SELECTION_PERCENTAGE
                    if (n_features * (FEATURE_SELECTION_PERCENTAGE / 100)) < 1:
                        actual_perc = 100 / n_features
                        print(f"    Adjusting feature selection to {actual_perc:.1f}% (minimum 1 feature)")
                    
                    feature_selector = SelectPercentile(percentile=actual_perc)
                    X_selected = feature_selector.fit_transform(X_scaled_df, train_target)
                    
                    # Step 4: Apply SMOTE using entire dataset (LEAKAGE)
                    if X_selected.shape[0] > k_neighbors:  # Only apply SMOTE if we have enough samples
                        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                        X_smote, y_smote = smote.fit_resample(X_selected, train_target)
                    else:
                        print(f"    Skipping SMOTE: not enough samples ({X_selected.shape[0]} <= {k_neighbors})")
                        X_smote, y_smote = X_selected, train_target
                    
                    # Step 5: Hyperparameter tuning using entire preprocessed dataset (LEAKAGE)
                    pipeline_leakage = Pipeline([
                        ('classifier', estimator_leakage)
                    ])
                    
                    param_grid = get_simplified_param_grid(alg)
                    
                    if param_grid:
                        # WORST CASE: Hyperparameter tuning on entire preprocessed dataset
                        grid_search_leakage = GridSearchCV(
                            pipeline_leakage, param_grid,
                            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=i),
                            scoring='accuracy', refit=True
                        )
                        
                        # Fit grid search on entire preprocessed dataset (MASSIVE LEAKAGE)
                        grid_search_leakage.fit(X_smote, y_smote)
                        
                        # Then evaluate the best model with CV (but data is already leaked)
                        cv_scores_leak_all.append(
                            make_dict_mean(
                                df_with_missing['target'].nunique(),
                                cross_validate(grid_search_leakage.best_estimator_, X_smote, y_smote, 
                                             cv=cv, scoring=scoring, return_train_score=False)
                            )
                        )
                    else:
                        # No hyperparameter tuning for this algorithm
                        cv_scores_leak_all.append(
                            make_dict_mean(
                                df_with_missing['target'].nunique(),
                                cross_validate(pipeline_leakage, X_smote, y_smote, 
                                             cv=cv, scoring=scoring, return_train_score=False)
                            )
                        )
                    
                    # Extract accuracy scores
                    cv_scores_noleak[i] = cv_scores_noleak_all[i]["test_accuracy"]
                    cv_scores_leak[i] = cv_scores_leak_all[i]["test_accuracy"]
                    
                    # Add metadata
                    param_grid = get_simplified_param_grid(alg)
                    for scores, leak_status in [(cv_scores_leak_all[i], True), (cv_scores_noleak_all[i], False)]:
                        scores["ds"] = ds
                        scores["iteration"] = i
                        scores["alg"] = alg
                        scores["leak"] = leak_status
                        scores["missing_percentage"] = MISSING_PERCENTAGE
                        scores["feature_selection_percentage"] = actual_perc
                        scores["original_features"] = n_features
                        scores["selected_features"] = X_selected.shape[1]
                        scores["final_samples"] = X_smote.shape[0] if 'X_smote' in locals() else X_selected.shape[0]
                        scores["minority_class_ratio"] = min(class_counts) / len(df_with_missing)
                        scores["hyperparameter_tuning"] = bool(param_grid)
                        scores["num_hyperparameters"] = len(param_grid) if param_grid else 0
                    
                    # Calculate leakage effect
                    leakage_effect = cv_scores_leak[i] - cv_scores_noleak[i]
                    
                    print(f"    Correct approach:   {cv_scores_noleak[i]:.4f}")
                    print(f"    Worst case leakage: {cv_scores_leak[i]:.4f}")
                    print(f"    Leakage effect:     {leakage_effect:+.4f}")
                    
                    # Combine results
                    df_temp = pd.DataFrame([cv_scores_leak_all[i], cv_scores_noleak_all[i]])
                    df_worst_case = pd.concat([df_worst_case, df_temp], ignore_index=True)
                    
                    # Save results after each iteration
                    df_worst_case.to_csv(f"worst_case_leakage_results{suffix}.csv", index=False)

        except Exception as e:
            print(f"FATAL ERROR processing dataset {ds}: {e}")
            print(f"Full traceback:")
            import traceback
            traceback.print_exc()
            raise e

    print("\nWorst case leakage testing completed!")
    
    # Print summary statistics
    if not df_worst_case.empty:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Calculate average leakage effect per algorithm
        for alg in target_algorithms:
            alg_data = df_worst_case[df_worst_case['alg'] == alg]
            if not alg_data.empty:
                leak_scores = alg_data[alg_data['leak'] == True]['test_accuracy']
                correct_scores = alg_data[alg_data['leak'] == False]['test_accuracy']
                
                if len(leak_scores) > 0 and len(correct_scores) > 0:
                    avg_leak = leak_scores.mean()
                    avg_correct = correct_scores.mean()
                    avg_effect = avg_leak - avg_correct
                    
                    print(f"{alg}:")
                    print(f"  Average correct approach:   {avg_correct:.4f}")
                    print(f"  Average worst case leakage: {avg_leak:.4f}")
                    print(f"  Average leakage effect:     {avg_effect:+.4f}")
                    print()
    
    return df_worst_case


if __name__ == "__main__":
    run_worst_case_leakage_test()
