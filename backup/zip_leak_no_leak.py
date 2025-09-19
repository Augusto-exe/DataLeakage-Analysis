import pandas as pd
from io import StringIO

ds_hyp = pd.read_csv("hyp.csv")
ds_imp = pd.read_csv("impute.csv")
ds_feat = pd.read_csv("feat_all.csv")
ds_norm = pd.read_csv("norm.csv")
#print(ds_norm.axes)
ds_norm = ds_norm.drop(['Unnamed: 0'], axis=1)
ds_hyp = ds_hyp.drop(['Unnamed: 0'], axis=1)
ds_feat = ds_feat.drop(['Unnamed: 0'], axis=1)
ds_imp = ds_imp.drop(['Unnamed: 0'], axis=1)

ds_imp = ds_imp[ds_imp['perc-miss'] == 0.1]
ds_imp = ds_imp.drop(['perc-miss'], axis=1)
ds_imp = ds_imp[ds_imp['imputer'] == "KNN"]
ds_imp = ds_imp.drop(['imputer'], axis=1)
ds_imp['task'] = 'impute'


ds_feat = ds_feat[ds_feat['perc'] == 10]
ds_feat = ds_feat.drop(['perc'], axis=1)
ds_feat['task'] = 'feat'

ds_hyp['task'] = 'hyperparameter'
ds_norm['task'] = 'normalization'


#print(ds_imp.axes)
#print(ds_feat)
#print(ds_norm.axes)
#print(ds_hyp.axes)pd.concat([data1, data2], axis=0) 

ds_all = pd.concat([ds_feat, ds_hyp], axis=0)
ds_all = pd.concat([ds_all, ds_imp], axis=0)
ds_all = pd.concat([ds_all, ds_norm], axis=0)
ds_all = ds_all.reset_index(drop=True)

# Sanity check: alternating True/False values
assert all(ds_all['leak'].iloc[::2].values) and not any(ds_all['leak'].iloc[1::2].values), "Data must alternate True/False"

# Split into True and False rows
true_rows = ds_all.iloc[::2].reset_index(drop=True)
false_rows = ds_all.iloc[1::2].reset_index(drop=True)

# Drop 'status' column from both (or keep if needed)
true_rows = true_rows.drop(columns=['leak'])
false_rows = false_rows.drop(columns=['leak'])

true_rows.to_csv("merged_leak_july.csv")
false_rows.to_csv("merged_no_leak_july.csv")

# Rename True row columns to have _leak suffix
true_rows.columns = [f"{col}_leak" for col in true_rows.columns]

# Concatenate horizontally (column-wise)
merged = pd.concat([false_rows, true_rows], axis=1)

print(merged)

#merged.to_csv("merged_results.csv")

