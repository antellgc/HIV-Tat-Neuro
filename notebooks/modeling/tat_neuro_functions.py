#!/usr/bin/env python

'''
This file defines functions for NGS Tat analysis pipeline
'''
from __future__ import division
import pandas as pd
import numpy as np
from collections import Counter
import itertools
from scipy import interp
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',family='Arial')

##################################################################

def plot_histogram(DF, feature, xlabel, savepath):
	"""Plots a histogram of GDS scores"""
	fig,ax = plt.subplots(figsize=(6,4))
	ax.hist(DF[feature])
	ax.set_xlabel(xlabel, size=16, fontweight='bold')
	ax.set_ylabel('Number of Samples', size=16, fontweight='bold')
	ax.set_axis_bgcolor('white')
	ax.spines['left'].set_visible(True)
	ax.spines['left'].set_color('black')
	ax.spines['left'].set_linewidth(2)
	ax.spines['bottom'].set_visible(True)
	ax.spines['bottom'].set_color('black')
	ax.spines['bottom'].set_linewidth(2)
	ax.axvline(x=0.5, ymin=0, ymax = 30, linewidth=2, color='k', ls='--')
	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14, length=5)
	ax.yaxis.tick_left()
	plt.show()
	fig.tight_layout()
	fig.savefig(savepath,dpi=300)

def split_df_by_impairment(DF, gds_score):
	"""creates two dataframes based on a GDS threshold"""
	GDS_mask = DF.GDS >= gds_score
	impaired_df = DF[GDS_mask]
	nonimpaired_df = DF[~GDS_mask]
	return impaired_df, nonimpaired_df

def plot_clinical_boxplots(DF, impaired_df, nonimpaired_df, savepath):
	"""creats boxplots of typical clinical parameters"""
	fig,ax = plt.subplots(2,3,figsize=(12,6))
	clin_cols = ['CD4','log10_VL','CD8','nCD4','log10_pVL','nCD8']
	plot_titles = ['CD4 count','Log10(Viral Load)','CD8 count',
					'nadir CD4','log10(peak VL)','nadir CD8']
	x,y,t = 0,0,0
	for feature in DF[clin_cols]:
		impaired_list = list(impaired_df[feature])
		nonimpaired_list = list(nonimpaired_df[feature])
		ax[x,y].tick_params(axis='x', labelsize=14)
		ax[x,y].tick_params(axis='y', labelsize=14, length=5)
		ax[x,y].yaxis.tick_left()
		ax[x,y].set_title(plot_titles[t], size=18)
		ax[x,y].set_xticklabels('')
		ax[x,y].boxplot([impaired_list, nonimpaired_list])
		# formatting
		ax[x,y].spines['left'].set_visible(True)
		ax[x,y].spines['left'].set_color('black')
		ax[x,y].spines['left'].set_linewidth(2)
		ax[x,y].spines['bottom'].set_visible(True)
		ax[x,y].spines['bottom'].set_color('black')
		ax[x,y].spines['bottom'].set_linewidth(2)
 		ax[x,y].set_axis_bgcolor('white')
		t += 1
		if x==1:
			ax[x,y].set_xticklabels(['Impaired', 'Nonimpaired'], size=16)
		else:
			ax[x,y].set_xticklabels(['', '']) 
		#update x and y coordinates
		if y!=2:
			y+=1
		else:
			y=0
			x+=1
	ax[0,1].set_ylabel('cells')
	plt.show()
	fig.tight_layout()
	fig.savefig(savepath, dpi=300)

####################################################################

def select_dataframe_columns(df, columns):
    '''Keeps only the relevant columns of the imported dataframe'''
    return df[columns]

def shuffle_dataframe(df):
    '''Shuffles the rows of the dataframe while preserving shape'''
    return df.sample(frac=1)

def target_feature_split(df, target, features):
    '''Splits target variable from model features'''
    feature_df = df[features]
    target = np.ravel(df[target])
    return target, feature_df

def scale_dataframe(df):
    '''Scales a dataframe containing float feature types'''
    scaled_array = scale(df)
    scaled_df = pd.DataFrame(scaled_array)
    scaled_df.columns = df.columns
    return scaled_df

def threshold_target(target, threshold):
    '''Converts a continuous target variable into a categorical variable'''
    return np.ravel([int(i>=threshold) for i in target])

def run_preparation_pipeline(df, features, target, threshold):
	'''executes pipeline of functions'''
	target, feature_df = target_feature_split(df, 'GDS', features)
	feature_df = pd.get_dummies(feature_df, drop_first=True)
	scaled_feature_df = scale_dataframe(feature_df)
	target2 = threshold_target(target, 0.5)
	X_df = scaled_feature_df
	y = target2
	return X_df, y

def run_preparation_pipeline2(df, features, target, threshold):
	'''executes pipeline of functions'''
	target, feature_df = target_feature_split(df, 'GDS', features)
	feature_df = pd.get_dummies(feature_df, drop_first=True)
	target2 = threshold_target(target, 0.5)
	X_df = feature_df
	y = target2
	return X_df, y

def run_genetic_preparation_pipeline(df, features, target, threshold):
	'''executes pipeline of functions'''
	target, feature_df = target_feature_split(df, 'GDS', features)
	feature_df = pd.get_dummies(feature_df, drop_first=True)
	target2 = threshold_target(target, 0.5)
	X_df = feature_df
	y = target2
	return X_df, y

####################################################################

def get_covariate_matrix(DF, features, target, threshold, savepath):
	label_dict = {'CD4':'CD4 count', 'Age':'Age','log10_VL':'log10 VL',
	'Gender_Male':'Gender','nCD8':'nadir CD8','CD8':'CD8 count',
	'nCD4':'nadir CD4', 'log10_pVL':'log10 peak VL'}
	X, y = run_preparation_pipeline(DF, features, target, threshold)
	X.columns = [label_dict[c] for c in X.columns if c in list(label_dict.keys())]
	plt.figure(figsize=(6, 6))
	sns.set(font_scale=1.5)
	sns.heatmap(X.corr(), square=True)
	plt.tight_layout()
	plt.savefig(savepath, dpi=300)
	plt.show()

##################################################################

def threshold_converter(old_df, new_df, column, threshold):
    t = old_df[column] > threshold
    new_df[column] = t.astype(int)
    return new_df

def make_threshold_dataframe(genetic_df, threshold):
    threshold_df = pd.DataFrame()
    for column in genetic_df.columns[1:]:
        threshold_df = threshold_converter(genetic_df, threshold_df, column, threshold)
    threshold_df['GDS'] = genetic_df['GDS']
    order = ['GDS'] + list(genetic_df.columns[1:])
    return threshold_df[order]

def get_consensus_df(g_df):
    all_max = []
    for i in range(1, g_df.shape[1], 20):
        pos_idx = range(i,i+20,1)
        pos_list = list(g_df.columns[pos_idx])
        pos_df = g_df[pos_list]
        all_max.append(list(pos_df.idxmax(axis=1)))
    consensus_dict = {}
    for L in all_max:
        p = L[0][:-1]
        consensus_dict[p] = []
        for variant in L:
            consensus_dict[p].append(variant[-1])
    df_new = pd.DataFrame(consensus_dict)
    df_new['GDS'] = g_df['GDS']
    sorted_cols = ['GDS'] + [str(i+1) for i in range(101)]
    return df_new[sorted_cols]

def get_pI_df(con_df):
	aa_pi_dict = {'A':6.00,'R':10.76,'N':5.41,'D':2.77,'C':5.07,'E':3.22,
              'Q':5.65,'G':5.97,'H':7.59,'I':6.02,'L':5.98,'K':9.74,
              'M':5.74,'F':5.48,'P':6.30,'S':5.68,'T':5.60,'W':5.89,
              'Y':5.66,'V':5.96}
	pI_df = con_df.replace(to_replace=aa_pi_dict)
	return pI_df

##################################################################

def get_CV_metrics(model, X, y, folds):
    '''Cross-Validation with many metrics'''
    roc_aucs, precisions, recalls, f1s = [], [], [], []
    FPRs, TPRs = [], []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    coefs = []
    # 5-fold cross-validation
    kf = StratifiedKFold(y, n_folds=folds)
    for train_index, test_index in kf:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        mod_prediction = model.fit(X_train, Y_train).predict(X_test)
        mod_probability = model.fit(X_train, Y_train).predict_proba(X_test)
        # metrics
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, mod_probability[:, 1])
        roc_auc = metrics.roc_auc_score(Y_test, mod_probability[:,1])
        precision = metrics.precision_score(Y_test, mod_prediction)
        recall = metrics.recall_score(Y_test, mod_prediction)
        f1 = metrics.f1_score(Y_test, mod_prediction)
        coef = model.coef_
        # append fold metrics
        FPRs.append(fpr)
        TPRs.append(tpr)
        roc_aucs.append(round(roc_auc,2))
        precisions.append(round(precision,2))
        recalls.append(round(recall,2))
        f1s.append(round(f1,2))
        coefs.append(coef)
        #coefs.append(coef[0])
        # mean calculations
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr /= folds
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    d = {'AUC':roc_aucs, 'Precision':precisions, 'Recall':recalls, 'F1':f1s, 'Coef':coefs,
         'FPRs':FPRs, 'TPRs':TPRs, 'meanTPR':mean_tpr, 'meanFPR':mean_fpr}
    return d

def get_RF_CV_metrics(model, X, y, folds):
    '''Cross-Validation with many metrics'''
    roc_aucs, precisions, recalls, f1s = [], [], [], []
    FPRs, TPRs = [], []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    # 5-fold cross-validation
    kf = StratifiedKFold(y, n_folds=folds)
    for train_index, test_index in kf:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        mod_prediction = model.fit(X_train, Y_train).predict(X_test)
        mod_probability = model.fit(X_train, Y_train).predict_proba(X_test)
        # metrics
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, mod_probability[:, 1])
        roc_auc = metrics.roc_auc_score(Y_test, mod_probability[:,1])
        precision = metrics.precision_score(Y_test, mod_prediction)
        recall = metrics.recall_score(Y_test, mod_prediction)
        f1 = metrics.f1_score(Y_test, mod_prediction)
        # append fold metrics
        FPRs.append(fpr)
        TPRs.append(tpr)
        roc_aucs.append(round(roc_auc,2))
        precisions.append(round(precision,2))
        recalls.append(round(recall,2))
        f1s.append(round(f1,2))
        #coefs.append(coef[0])
        # mean calculations
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr /= folds
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    d = {'AUC':roc_aucs, 'Precision':precisions, 'Recall':recalls, 'F1':f1s,
         'FPRs':FPRs, 'TPRs':TPRs, 'meanTPR':mean_tpr, 'meanFPR':mean_fpr}
    return d

def metrics_wrapper(df, model, f, folds):
    target = 'GDS'
    threshold = 0.5
    features = ['Age','Gender','log10_VL','log10_pVL','CD4','nCD4','CD8','nCD8','TMHDS']
    df_shuffled = shuffle_dataframe(df)
    X_df, y = run_preparation_pipeline(df_shuffled, features, target, threshold)
    d = get_CV_metrics(model, X_df[f], y, folds)
    return d

def metrics_wrapper_random(df, model, f, folds):
    target = 'GDS'
    threshold = 0.5
    features = ['Age','Gender','log10_VL','log10_pVL','CD4','nCD4','CD8','nCD8','TMHDS']
    df_shuffled = shuffle_dataframe(df)
    X_df, y = run_preparation_pipeline(df_shuffled, features, target, threshold)
    y_rand = np.random.permutation(y)
    d = get_CV_metrics(model, X_df[f], y_rand, folds)
    return d

####################################################################
def filter_genetic_variants(genetic_df, a, b):
    keep_cols = ['GDS']
    for col in genetic_df.columns[1:]:
        s = sum(genetic_df[col] > a)
        L = len(genetic_df[col])
        passed = (s/L) > b
        if passed:
            #print (col, round(s/L,2))
            keep_cols.append(col)
	bad_cols = genetic_df.columns[1:61]
    keep_cols = [x for x in keep_cols if x not in bad_cols]
    return genetic_df[keep_cols]

