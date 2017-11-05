# Python functions for processing AA count data preprocessing
# Converts file formats into frequency or count data tables 
#
# Developed by: Gregory Antell
# Developed Date: 12-Dec-2016
# Latest Revision Date: 4-Nov-2017

from __future__ import division
import glob
import pandas as pd
import numpy as np
import itertools
from collections import Counter
import time
import matplotlib.pyplot as plt

def tsvfile_to_df(tsv_file, protein):
    '''
    Converts an amino acid count tsv_file into a pandas dataframe
    '''
    patient = tsv_file.split('/')[-1].split('.')[0].split('-')[0]
    visit = tsv_file.split('/')[-1].split('.')[0].split('-')[1]
    df = pd.read_csv(tsv_file, delimiter='\t')
    df = df[df.Prot == protein].reset_index()
    df['AAPos'] = df['Pos']+1
    df['Patient'] = patient
    df['Visit'] = visit
    order = ['Patient','Visit','Prot','AAPos','AA','Count']
    df = df[order]
    return df

def remove_stops(df):
    '''
    Removes all * characters from dataframe and returns
    a new dataframe
    '''
    nogaps_df = df[df['AA']!='*']
    return nogaps_df

def makeFreqDict(df):
    '''
    Creates a dictionary storing the frequency (as a percentage value) for
    each amino acid at each position of the protein
    '''
    #intialize dictionary
    aminoacids = 'ARNDCQEGHILKMFPSTWYV'
    freq_dict = {'Patient':[], 'Visit':[], 'Prot':[], 'AAPos':[], 'Coverage':[],
                 'A':[],'R':[],'N':[],'D':[],'C':[],'Q':[],'E':[],'G':[],'H':[],'I':[],
                 'L':[],'K':[],'M':[],'F':[],'P':[],'S':[],'T':[],'W':[],'Y':[],'V':[],}

    for i, group in df.groupby('AAPos'):  
        #convert counts to percentages
        total = sum(group['Count'])
        group['Percent'] = group['Count']/total
        #print group
        #get values for each position
        patient = list(group['Patient'])[0]
        visit = list(group['Visit'])[0]
        orf = list(group['Prot'])[0]
        pos = list(group['AAPos'])[0]
        cov = total
        #fill up dictionary with values
        freq_dict['Patient'].append(patient)
        freq_dict['Visit'].append(visit)
        freq_dict['Prot'].append(orf)
        freq_dict['AAPos'].append(pos)
        freq_dict['Coverage'].append(cov)  
        #fill up dictionary with frequencies
        for aa in aminoacids:
            x = group[group['AA'] == aa]
            if x.shape[0] == 0:
                freq_dict[aa].append(0)
            else:
                freq_dict[aa].append(list(x['Percent'])[0])
    return freq_dict

def makeCountDict(df):
    '''
    Creates a dictionary storing the frequency (as an integer value) for
    each amino acid at each position of the protein
    '''
    #intialize dictionary
    aminoacids = 'ARNDCQEGHILKMFPSTWYV'
    count_dict = {'Patient':[], 'Visit':[], 'Prot':[], 'AAPos':[], 'Coverage':[],
                 'A':[],'R':[],'N':[],'D':[],'C':[],'Q':[],'E':[],'G':[],'H':[],'I':[],
                 'L':[],'K':[],'M':[],'F':[],'P':[],'S':[],'T':[],'W':[],'Y':[],'V':[],}
    for i, group in df.groupby('AAPos'):
        total = sum(group['Count'])
        #get values for each position
        patient = list(group['Patient'])[0]
        visit = list(group['Visit'])[0]
        orf = list(group['Prot'])[0]
        pos = list(group['AAPos'])[0]
        cov = total
        #fill up dictionary with values
        count_dict['Patient'].append(patient)
        count_dict['Visit'].append(visit)
        count_dict['Prot'].append(orf)
        count_dict['AAPos'].append(pos)
        count_dict['Coverage'].append(cov)  
        #fill up dictionary with frequencies
        for aa in aminoacids:
            x = group[group['AA'] == aa]
            if x.shape[0] == 0:
                count_dict[aa].append(0)
            else:
                count_dict[aa].append(list(x['Count'])[0])
    return count_dict

def freqToDataframe(freq_dict):
    '''
    Converts dictionary to a dataframe with the proper column order
    '''
    col_order = ['Patient','Visit','Prot','AAPos','Coverage',
             'A','R','N','D','C','Q','E','G','H','I',
             'L','K','M','F','P','S','T','W','Y','V']
    freq_df = pd.DataFrame(freq_dict)[col_order]
    return freq_df