import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

alg_list = ["KNN", "SVC", "LR", "NB","DT", "RF"]
scoring = ["test_accuracy","test_balanced_accuracy","test_f1","test_precision","test_recall"]
ds_list = ["GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1","agaricus_lepiota","mushroom","ring","twonorm","clean1","dna","phoneme","mfeat_pixel","banana","mfeat_factors","spambase","Hill_Valley_with_noise","Hill_Valley_without_noise","waveform_40","waveform_21","movement_libras","satimage","chess","kr_vs_kp","optdigits","splice","texture","sonar","molecular_biology_promoters","mfeat_fourier","analcatdata_authorship","tokyo1","soybean","mfeat_karhunen"]

ds_hyp = pd.read_csv("hyp.csv")
ds_imp = pd.read_csv("impute.csv")
ds_feat = pd.read_csv("feat.csv")
ds_norm = pd.read_csv("norm.csv")
ds_hyp._append(ds_feat)
ds_hyp._append(ds_norm)
ds_hyp._append(ds_imp)

cv_scores_leak = {}
cv_scores_noleak = {}
difference = {} 
diffs =[]
ds ='a'
alg = 'a'
#for ds in ds_list:
cv_scores_leak[ds] = {}
cv_scores_noleak[ds]= {}
difference[ds] = {} 

#for alg in alg_list:
cv_scores_noleak[ds][alg] = []
cv_scores_leak[ds][alg] = []
difference[ds][alg] = []
for it in range(0,6):
#for perc in perc_select:
  df_leak = ds_hyp[ds_hyp['leak'] == True]
  df_leak =df_leak [df_leak['iteration'] == it]
  #df_leak = df_leak[df_leak['ds'] == ds]
  #df_leak = df_leak[df_leak['alg'] == alg]

  #df_alg_leak =df_alg_leak [df_alg_leak ['ds'] == ds]
  #df_alg_leak =df_alg_leak [df_alg_leak ['perc'] == perc]
  avg_alg_leak_acc = df_leak['test_balanced_accuracy'].mean()
  cv_scores_leak[ds][alg].append(avg_alg_leak_acc)

  df_no_leak = ds_hyp[ds_hyp['leak'] == False]
  df_no_leak =df_no_leak [df_no_leak ['iteration'] == it]
  #df_no_leak = df_no_leak[df_no_leak['ds'] == ds]
  #df_no_leak = df_no_leak[df_no_leak['alg'] == alg]
  #df_alg_no_leak =df_no_leak [df_no_leak ['ds'] == ds]
  #df_alg_no_leak =df_alg_no_leak [df_alg_no_leak ['perc'] == perc]
  avg_alg_no_leak_acc = df_no_leak['test_balanced_accuracy'].mean()
  cv_scores_noleak[ds][alg].append(avg_alg_no_leak_acc)

  diff = avg_alg_leak_acc-avg_alg_no_leak_acc
  difference[ds][alg].append(diff)
      

# Create the figure and axis objects
ds_list = ['a']
alg_list = ds_list
for ds in ds_list:
    for alg in alg_list:
      
      fig, ax1 = plt.subplots()
      data = [cv_scores_leak[ds][alg], cv_scores_noleak[ds][alg]]  # Combine the two datasets into a single list

      plt.boxplot(data)
      plt.xticks([1, 2], ['Leak', 'No Leak'])  # Customize the x-axis labels
      plt.xlabel('Datasets')
      plt.ylabel('Values')
      plt.title('Box Plot ' + alg + ' ' + ds)

      #img = fig2img(fig)
      #img.save('imgs/box_plots_f1/'+alg+'_'+ds+'.png')
      #plt.close()

      plt.show()
      '''
      fig, ax1 = plt.subplots()
      fig.suptitle(alg + ' - '+ds, fontsize=16)

      # Plot the scores as lines, using the first axis
      ax1.plot(cv_scores_noleak[ds][alg], label='Scores no Leak' , color='blue')
      ax1.plot(cv_scores_leak[ds][alg], label='Scores Leak', color='red')

      # Set the y-axis label for the scores
      ax1.set_ylabel('Score', color='black')
      ax1.tick_params(axis='y', labelcolor='black')

      # Create a second y-axis for the difference
      ax2 = ax1.twinx()

      # Plot the difference as a bar, using the second axis
      ax2.bar(np.arange(len(difference[ds][alg])), difference[ds][alg], alpha=0.5, color='green')

      # Set the y-axis label for the difference
      ax2.set_ylabel('Difference', color='green')
      ax2.tick_params(axis='y', labelcolor='green')

      # Add a legend to the plot
      ax1.legend()


      img = fig2img(fig)
      img.save('imgs/balanced/'+alg+'_'+ds+'.png')
      plt.close()
      #plt.show()
      '''
'''
fig, ax1 = plt.subplots()
data = [diffs]  # Combine the two datasets into a single list

plt.boxplot(data)
plt.xticks([1], ['diff'])  # Customize the x-axis labels
plt.xlabel('Datasets')
plt.ylabel('Values')
plt.title('Box Plot ')

plt.show()
'''