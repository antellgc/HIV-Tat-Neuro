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
%matplotlib inline

def import_redcap_data(redcap_path):
    '''
    Imports REDCAP .cvs-formatted data dump
    '''
    redcap_df = pd.read_csv(redcap_path, sep='\t')
    redcap_df.head()
    return redcap_df

def select_clinical_parameters(df, cols):
    '''
    Choose the columns to keep as relevant clinical parameters
    '''
    clin_df = df[cols.keys()].rename(columns=cols)
    clin_df['DateOfVisit'] = pd.to_datetime(clin_df['DateOfVisit'])
    return clin_df