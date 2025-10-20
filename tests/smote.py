import sys
sys.path.append('..')  # Adjust the path as needed
from model_utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def run_smote_test():
    print("SMOTE (Synthetic Minority Oversampling Technique)")
    target_ds_list = ds_list
    print("Datasets to be used: ", target_ds_list)

    df_smote = pd.DataFrame()
    
    # Define imbalance scenarios to test
    imbalance_scenarios = [
        {'name': 'natural', 'ratios': None, 'min_ratio_threshold': 0},  # Use natural distribution, always run
    ]
    
    for scenario in imbalance_scenarios:
        print(f"\n=== TESTING SCENARIO: {scenario['name']} ===")
        
        for ds in target_ds_list:
            print(f"\n TESTING ds: _ {ds} _ with scenario: {scenario['name']}\n")
            
            try:
                df_original = fetch_data(ds)
                
                # Skip if dataset has only one class
                original_class_counts = df_original['target'].value_counts()
                if len(original_class_counts) < 2:
                    print(f"Skipping {ds}: Only one class present")
                    continue
                
                # Handle different scenarios
                if scenario['name'] == 'natural':
                    # Use natural distribution, always run regardless of balance
                    min_class_ratio = original_class_counts.min() / len(df_original)
                    print(f"Using natural distribution (min class ratio: {min_class_ratio:.3f})")
                    df = df_original.copy()
                else:
                    # This block is not used with current scenarios but kept for future extension
                    df = df_original.copy()
                
                class_counts = df['target'].value_counts()
                min_class_ratio = class_counts.min() / len(df)
                
                print(f"Class distribution: {dict(class_counts)}")
                print(f"Minority class ratio: {min_class_ratio:.3f}")
                
                scoring = get_scoring(df['target'].nunique())
                target_algorithms = new_algorithms
                
                for alg in target_algorithms:
                    print(f"alg: {alg}")
                    estimator = get_estimator(alg)

                    # Arrays to store scores
                    cv_scores_leak = np.zeros(NUM_TRIALS)
                    cv_scores_leak_all = []
                    cv_scores_noleak = np.zeros(NUM_TRIALS)
                    cv_scores_noleak_all = []

                    # Loop for each trial
                    for i in range(NUM_TRIALS):
                        train_data = df.drop('target', axis=1)
                        train_target = df['target']

                        # Calculate safe k_neighbors for SMOTE (must be less than minority class size)
                        min_class_size = min(class_counts)
                        k_neighbors = min(5, max(1, min_class_size - 1))
                        
                        # If minority class is too small for SMOTE, use k_neighbors=1
                        if k_neighbors < 1:
                            k_neighbors = 1
                            print(f"Warning: Very small minority class ({min_class_size}), using k_neighbors=1")
                        else:
                            print(f"Using k_neighbors={k_neighbors} for minority class size={min_class_size}")

                        # NO LEAKAGE: SMOTE applied within cross-validation
                        # Create pipeline with SMOTE inside (correct approach)
                        pipeline_no_leak = ImbPipeline([
                            ('scaler', StandardScaler()),
                            ('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)),
                            ('classifier', estimator)
                        ])
                        
                        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
                    
                    # No leakage: SMOTE applied within each CV fold
                    cv_scores_noleak_all.append(
                        make_dict_mean(
                            df['target'].nunique(),
                            cross_validate(pipeline_no_leak, train_data, train_target, 
                                         cv=cv, scoring=scoring, return_train_score=False)
                        )
                    )

                    # WITH LEAKAGE: SMOTE applied to entire dataset first (incorrect approach)
                    # Apply SMOTE to the entire dataset (data leakage)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(train_data)
                    
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_smote, y_smote = smote.fit_resample(X_scaled, train_target)
                    
                    # Convert back to DataFrame for consistency
                    X_smote_df = pd.DataFrame(X_smote, columns=train_data.columns)
                    
                    # Now apply cross-validation on the SMOTE-processed data
                    pipeline_leak = Pipeline([
                        ('classifier', estimator)
                    ])
                    
                    cv_scores_leak_all.append(
                        make_dict_mean(
                            df['target'].nunique(),
                            cross_validate(pipeline_leak, X_smote_df, y_smote, 
                                         cv=cv, scoring=scoring, return_train_score=False)
                        )
                    )

                    # Extract accuracy scores
                    cv_scores_noleak[i] = cv_scores_noleak_all[i]["test_accuracy"]
                    cv_scores_leak[i] = cv_scores_leak_all[i]["test_accuracy"]

                    # Add metadata
                    cv_scores_leak_all[i]["ds"] = ds
                    cv_scores_leak_all[i]["iteration"] = i
                    cv_scores_leak_all[i]["alg"] = alg
                    cv_scores_leak_all[i]["leak"] = True
                    cv_scores_leak_all[i]["minority_class_ratio"] = min_class_ratio
                    cv_scores_leak_all[i]["imbalance_scenario"] = scenario['name']

                    cv_scores_noleak_all[i]["ds"] = ds
                    cv_scores_noleak_all[i]["iteration"] = i
                    cv_scores_noleak_all[i]["alg"] = alg
                    cv_scores_noleak_all[i]["leak"] = False
                    cv_scores_noleak_all[i]["minority_class_ratio"] = min_class_ratio
                    cv_scores_noleak_all[i]["imbalance_scenario"] = scenario['name']

                    # Combine results
                    df_temp = pd.DataFrame([cv_scores_leak_all[i], cv_scores_noleak_all[i]])
                    df_smote = pd.concat([df_smote, df_temp], ignore_index=True)

                    print(f"CV scores (with data leakage) {alg} in {ds}: {cv_scores_leak[i]:.4f}")
                    print(f"CV scores (without data leakage) {alg} in {ds}: {cv_scores_noleak[i]:.4f}")
                    
                    # Save results after each iteration
                    df_smote.to_csv(f"smote_results{suffix}.csv", index=False)

            except Exception as e:
                print(f"FATAL ERROR processing dataset {ds} with scenario {scenario['name']}: {e}")
                print(f"Full traceback:")
                import traceback
                traceback.print_exc()
                raise e

            except Exception as e:
                print(f"Error processing dataset {ds} with scenario {scenario['name']}: {e}")
                continue

    print("SMOTE testing completed!")
    return df_smote


if __name__ == "__main__":
    run_smote_test()
