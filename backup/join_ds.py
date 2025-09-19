import pandas as pd


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


print(ds_all)
print(ds_all.axes)
scoring = ["test_accuracy","test_balanced_accuracy","test_f1","test_precision","test_recall"]
for column in scoring:
  ds_all[column+'_difference'] = ds_all.groupby(ds_all.index // 2)[column].diff()
  
ds_all = ds_all.dropna(subset=[column+'_difference' for column in scoring])
ds_all = ds_all.drop(scoring, axis=1)
ds_all = ds_all.rename(columns={column+'_difference': column for column in scoring})
ds_all = ds_all.reset_index(drop=True)


ds_all = ds_all.drop(['iteration'], axis=1)
ds_all = ds_all.drop(['leak'], axis=1)

print(ds_all)
print(ds_all.axes)

ds_all.to_csv('all_values.csv')
#selected_data = pd.read_csv('expanded.csv')
#selected_data = selected_data.rename(columns={'Dataset': 'ds'})

#ds_all = ds_all.merge(selected_data, on='ds', how='left')

#ds_all.to_csv('all2.csv')
