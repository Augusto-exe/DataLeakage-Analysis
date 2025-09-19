import pandas as pd

# # Read the CSV file
# feat_LR = pd.read_csv('consolidated_results_21_04_25/feat_LR.csv', header=None)
# feat_LR = feat_LR.drop(columns=[0])
# print(feat_LR.head())
# print(feat_LR.shape)  # This will show the number of rows and columns
# # Assign column names for clarity
# feat_LR.columns = [
#     'test_accuracy', 'test_balanced_accuracy', 'test_f1', 'test_precision', 'test_recall',
#     'ds', 'task', 'alg', 'leak', 'perc'
# ]

# # Separate the rows into "no leak" (TRUE) and "leak" (FALSE)
# no_leak = feat_LR[feat_LR['leak'] == 'TRUE']
# leak = feat_LR[feat_LR['leak'] == 'FALSE']

# # Ensure the rows are aligned by 'ds', 'task', and 'alg'
# merged = pd.merge(no_leak, leak, on=['ds', 'task', 'alg'], suffixes=('_no_leak', '_leak'))

# # Calculate the differences for the specified columns
# merged['test_accuracy'] = merged['test_accuracy_no_leak'] - merged['test_accuracy_leak']
# merged['test_balanced_accuracy'] = merged['test_balanced_accuracy_no_leak'] - merged['test_balanced_accuracy_leak']
# merged['test_f1'] = merged['test_f1_no_leak'] - merged['test_f1_leak']
# merged['test_precision'] = merged['test_precision_no_leak'] - merged['test_precision_leak']
# merged['test_recall'] = merged['test_recall_no_leak'] - merged['test_recall_leak']

# # Create the final DataFrame with the required columns
# result = merged[[
#     'ds', 'alg', 'task',
#     'test_accuracy', 'test_balanced_accuracy', 'test_f1',
#     'test_precision', 'test_recall'
# ]]

# # Save the result to a new CSV file
# result.to_csv('feat_LR_diff.csv', index=False)


# I janto to concatenate the values from the feat_LR.csv into the end of the file all.csv and make a all_new.csv file
# import pandas as pd
# Read the CSV files
feat_LR = pd.read_csv('consolidated_results_21_04_25/feat.csv')
all_data = pd.read_csv('feat_LR_all.csv')
# Concatenate the dataframes
all_new = pd.concat([all_data, feat_LR], axis=0)
# Save the new dataframe to a CSV file
all_new.to_csv('feat_all.csv', index=False)
