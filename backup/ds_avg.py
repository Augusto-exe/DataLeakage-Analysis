import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import scipy.stats as stats
import os

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
perc_list = [0.05,0.1,0.2,0.3]
perc_select = [1,5,10,20]
std_perc_miss = 0.2
std_perc_select = 10
alg_list = ["KNN", "SVC", "LR", "NB","DT", "RF"]
scoring = ["test_accuracy","test_balanced_accuracy","test_f1","test_precision","test_recall"]
ds_list = ["GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1","agaricus_lepiota","mushroom","ring","twonorm","clean1","dna","phoneme","mfeat_pixel","banana","mfeat_factors","spambase","Hill_Valley_with_noise","Hill_Valley_without_noise","waveform_40","waveform_21","movement_libras","satimage","chess","kr_vs_kp","optdigits","splice","texture","sonar","molecular_biology_promoters","mfeat_fourier","analcatdata_authorship","tokyo1","soybean","mfeat_karhunen"]
meta_list = ["n_observations",	"n_features",	"n_classes",	"Imbalance",	"feat_class",	"inst_class",	"feat_obs"]
meta_name = ["Inst.","Feats_Bin.","Feats_Categ.","Feats_Cont.","Feats","Classes","Imbalance","Feat_Class","Inst_Class","Feat_Inst"]
ds_hyp = pd.read_csv("hyp.csv")
ds_imp = pd.read_csv("impute.csv")
ds_feat = pd.read_csv("feat_all.csv")
ds_norm = pd.read_csv("norm.csv")
ds_general = pd.read_csv("hyp.csv")
ds_general = pd.concat([ds_general, ds_imp[ds_imp["perc-miss"]== std_perc_miss]], ignore_index=True)
ds_general = pd.concat([ds_general, ds_feat[ds_feat["perc"]==std_perc_select]], ignore_index=True)
ds_general = pd.concat([ds_general, ds_norm], ignore_index=True)
ds_dict = {'hyp':ds_hyp,'imp':ds_imp,'feat':ds_feat,'norm':ds_norm,'general':ds_general}
feat_list = ['hyp','imp','feat','norm','general']

#ds_hyp = ds_norm
#ds_meta = pd.read_csv("expanded.csv")
#print(ds_meta)
cv_scores_leak = {}
cv_scores_noleak = {}
difference = {} 
diffs =[]
output_dir = 'imgs/algs_bp'
os.makedirs(output_dir, exist_ok=True)

#print the lenght of final ds


#for alg in alg_list:
cv_scores_leak = {}
cv_scores_noleak= {}
difference = {}

plt.rcParams.update({'font.size': 13})

#for alg in alg_list:
  #cv_scores_noleak[ds][alg] = []
  #cv_scores_leak[ds][alg] = []
  #difference[ds][alg] = 
#for it in range(0,6):
#for perc in perc_select:
#for alg in alg_list:

# for alg in alg_list:
#   cv_scores_leak[alg] ={}
#   cv_scores_noleak[alg] = {}
#   difference[alg] = {}
diffs = {} 
#meta_diff ={"Inst." : [],"Feats_Bin.": [],"Feats_Categ.": [],"Feats_Cont.": [],"Feats" : [],"Classes": [],"Imbalance" : [],"Feat_Class" : [],"Inst_Class" : [],"Feat_Inst": []}
#for feat in feat_list:
meta_diff ={}
#for feat in feat_list:
feat = "general"
meta_diff[feat] = {}
diffs[feat] = {}
cv_scores_leak[feat] ={}
cv_scores_noleak[feat] = {}
difference[feat] = {}
for metric in ['balanced_accuracy','f1']:
  meta_diff[feat][metric] = {}
  #meta_diff[feat][metric] = []
  diffs[feat][metric] = {}
  cv_scores_leak[feat][metric] ={}
  cv_scores_noleak[feat][metric] = {}
  difference[feat][metric] = {}
  for alg in alg_list:
    meta_diff[feat][metric][alg] =  {"Inst." : [],"Feats_Bin.": [],"Feats_Categ.": [],"Feats_Cont.": [],"Feats" : [],"Classes": [],"Imbalance" : [],"Feat_Class" : [],"Inst_Class" : [],"Feat_Inst": []}
    diffs[feat][metric][alg]= []
    cv_scores_leak[feat][metric][alg] =[]
    cv_scores_noleak[feat][metric][alg] = []
    difference[feat][metric][alg] = []
    for it in range(0,6):
      df_leak = ds_dict[feat][ds_dict[feat]['leak'] == True]
      df_leak =df_leak [df_leak['iteration'] == it]
      # df_leak = df_leak[df_leak['ds'] == ds]
      #df_leak = df_leak[df_leak['imputer'] == imputer]
      #df_leak = df_leak[df_leak['perc'] == percmiss]
      df_leak = df_leak[df_leak['alg'] == alg]

      avg_alg_leak_acc = df_leak['test_'+metric].mean()*100
      cv_scores_leak[feat][metric][alg].append(avg_alg_leak_acc)
      #cv_scores_leak[alg][metric]= (avg_alg_leak_acc)
      #cv_scores_leak.append(avg_alg_leak_acc)

      df_no_leak = ds_dict[feat][ds_dict[feat]['leak'] == False]
      df_no_leak =df_no_leak [df_no_leak ['iteration'] == it]
      # df_no_leak = df_no_leak[df_no_leak['ds'] == ds]
      #df_no_leak = df_no_leak[df_no_leak['imputer'] == imputer]
      #df_no_leak = df_no_leak[df_no_leak['perc'] == percmiss]
      df_no_leak = df_no_leak[df_no_leak['alg'] == alg]
      
      avg_alg_no_leak_acc = df_no_leak['test_'+metric].mean()*100
      cv_scores_noleak[feat][metric][alg].append(avg_alg_no_leak_acc)
      #cv_scores_noleak[alg][metric]= (avg_alg_no_leak_acc)

      #cv_scores_noleak.append(avg_alg_no_leak_acc)

      diff = avg_alg_leak_acc-avg_alg_no_leak_acc
      difference[feat][metric][alg].append(diff)
      #print(df_leak['test_'+metric])
      '''
      leak_res = []
      no_leak_res = []
      for val in df_leak['test_'+metric]:
        leak_res.append(val*100)
      for val in df_no_leak['test_'+metric]:
        no_leak_res.append(val*100)
      for i in range(len(leak_res)):
        diffs[feat][metric][imputer].append(leak_res[i]-no_leak_res[i])
        #meta_diff[feat][metric].append(perc)
        for meta in meta_name:
          meta_diff[feat][metric][imputer][meta].append(ds_meta[ds_meta["Dataset"] == ds][meta].item())
      '''
      
    #print(difference[metric][perc])
'''
for feat in ["imp"]:
  print(feat+":")
  for metric in ['balanced_accuracy','f1']:
    print(">"+metric+":")
    for imputer in ['mean','median','KNN']:
      print(">>"+imputer+":")
      for metadata in meta_name:
        tau, p_value = stats.kendalltau(diffs[feat][metric][imputer], (meta_diff[feat][metric][imputer][metadata] ))
        #slope, intercept, r_value, p_value, std_err = stats.linregress((meta_diff[feat][metric] ), diffs[feat][metric])
        print(">>>"+metadata+": ",tau)
        fig, ax1 = plt.subplots() 
        plt.plot((meta_diff[feat][metric][imputer][metadata] ), diffs[feat][metric][imputer], 'o', color='red')
        #plt.plot((meta_diff[feat][metric] ), slope * np.array((meta_diff[feat][metric] )) + intercept, 'b-', label='Regression Line')
        plt.xlabel(metadata)
        plt.ylabel('Difference')
        #plt.title('Difference x ')
        #plt.grid(True)
        plt.tight_layout()
        img = fig2img(fig)
        img.save('imgs/test/new_'+feat+'_'+imputer+'_'+metric+'_'+metadata+'.png')
        #plt.show()
        plt.close()
'''


box_colors = ['lightcoral', 'lightblue']
for alg in alg_list:
  for metric in ['balanced_accuracy', 'f1']:
    data = [cv_scores_leak[feat][metric][alg], cv_scores_noleak[feat][metric][alg]]
    tags = ["With leakage", "Without leakage"]

    fig, ax1 = plt.subplots()

    # Create a boxplot with customized box colors
    bplot = ax1.boxplot(data, patch_artist=True, vert=True)

    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
            median.set_color('black')

    plt.xticks(range(1, len(tags) + 1), tags)
    plt.xlabel('Test')
    plt.ylabel('Scores')
    # plt.title(f'Box Plot for {alg} - {metric}')
    plt.tight_layout()
    img = fig2img(fig)
    img.save('imgs/algs_bp/NEW_title_box_plot_'+alg+'_'+metric+".png")
    plt.close()
# Create the figure and axis objects
# for feat in feat_list:
#   for metric in ['balanced_accuracy', 'f1']:
#     data = [cv_scores_leak[feat][metric], cv_scores_noleak[feat][metric]]
#     tags = ["With DL", "Without DL"]

#     fig, ax1 = plt.subplots()

#     # Customizing boxplot properties, including box color
#     boxprops = dict(facecolor='lightgray', color='black')
#     bplot = ax1.boxplot(data, boxprops=boxprops)

#     plt.xticks(range(1, len(tags) + 1), tags)
#     plt.xlabel('Test')
#     plt.ylabel('Scores')
#     plt.tight_layout()
#     img = fig2img(fig)
#     img.save('imgs/box_plot_'+feat+'_'+metric+".png")
#     plt.close()


# for ds in ds_list:
#   for feat in ['general']: 
#     for metric in ['balanced_accuracy','f1']:
#       #    for feat in feat_list:
#       fig, ax1 = plt.subplots()
#       #fig.suptitle("general diff", fontsize=16)

#       # Plot the scores as lines, using the first axis
#       ax1.plot(cv_scores_noleak[feat][metric][ds], label='Performance without leakage' , color='blue')
#       ax1.plot(cv_scores_leak[feat][metric][ds], label='Performance with leakage', color='red')

#       # Set the y-axis label for the scores
#       ax1.set_ylabel('Performance', color='black')
#       ax1.set_xlabel('Repetition', color='black')
#       ax1.tick_params(axis='y', labelcolor='black')
#       ax1.tick_params(axis='x', labelcolor='black')

#       # Create a second y-axis for the difference
#       ax2 = ax1.twinx()

#       # Plot the difference as a bar, using the second axis
#       ax2.bar(np.arange(len(difference[feat][metric][ds])), difference[feat][metric][ds], alpha=0.5, color='green')

#       # Set the y-axis label for the difference
#       ax2.set_ylabel('Difference', color='green')
#       ax2.tick_params(axis='y', labelcolor='green')

#       #Add a legend to the plot
#       ax1.legend()

#       plt.tight_layout()

#       img = fig2img(fig)
#       # img.save('a.png')
#       plt.savefig('imgs/ds_diff_new/NEW_'+ds+'_'+metric+'_diff.png',bbox_inches='tight')
      
#       #plt.show()
#       plt.close()

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


      #img = fig2img(fig)
      #img.save('imgs/balanced/'+alg+'_'+ds+'.png')
      #plt.close()
      plt.show()
      

fig, ax1 = plt.subplots()
data = [diffs]  # Combine the two datasets into a single list

plt.boxplot(data)
plt.xticks([1], ['diff'])  # Customize the x-axis labels
plt.xlabel('Datasets')
plt.ylabel('Values')
plt.title('Box Plot ')

plt.show()
'''