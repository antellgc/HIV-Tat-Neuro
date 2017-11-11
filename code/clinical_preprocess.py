# Python functions for clinical data preprocessing
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

def import_redcap_data(redcap_path):
    '''Imports REDCAP .cvs-formatted data dump'''
    redcap_df = pd.read_csv(redcap_path, sep='\t', low_memory=False)
    redcap_df.head()
    return redcap_df

def select_clinical_parameters(df, cols):
    '''Choose the columns to keep as relevant clinical parameters'''
    clin_df = df[list(cols.keys())].rename(columns=cols)
    clin_df['DateOfVisit'] = pd.to_datetime(clin_df['DateOfVisit'])
    return clin_df

def update_gds_data(GDS_df, clin_df):
    visits = []
    for idx, row in GDS_df.iterrows():
        NEURO_PATIENT = row['PatientID']
        NEURO_DATE = row['VisitDate']    
        clin_info = clin_df[clin_df.Patient == NEURO_PATIENT][['Patient', 'Visit', 'DateOfVisit']]
        clin_info2 = clin_info[clin_info.DateOfVisit == NEURO_DATE][['Patient', 'Visit', 'DateOfVisit']]
        if clin_info2.empty:
            for a, b in clin_info.dropna().iterrows():
                days_difference = b.DateOfVisit - NEURO_DATE
                if abs(days_difference.days) < 7:
                    visits.append(b.Visit)
        else:
            visits.append(list(clin_info2['Visit'])[0]) 
    GDS_df['Visit'] = visits
    return GDS_df[['PatientID', 'Visit', 'VisitDate', 'GDS']]

merged_df = pd.merge(clinical_df, seq_abundance_df,
                  left_on = ['Patient','Visit'],
                  right_on = ['Patient','Visit'],
                  how = 'inner')
merged_df1.sort_values(['AAPos','Patient','Visit'],  inplace=True)
print merged_df1.shape
