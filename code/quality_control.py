# Python functions for NGS data quality control
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
import matplotlib as mpl
import matplotlib.pyplot as plt

def filter_freq_files(wildcard_path, n):
    """
    require at least 100 read coverage for 80% of the
    total positions, n, in the genomic region 
    """
    freq_files = glob.glob(wildcard_path)
    quality_freq_files = []
    passed = 0
    for freq_file in freq_files:
        df = pd.read_csv(freq_file)
        q = sum(df['Coverage'] > 100) / n
        if q >= 0.8:
            #print freq_file.split('/')[-1].split('.')[0],
            passed+=1
            quality_freq_files.append(freq_file)
    print(passed, 'files passed QC filter')
    return quality_freq_files

def drop_positions(df, position):
    df = df[df['AAPos'] <= position]
    return df

def combine_quality_files(quality_files, position):
    freq_files = quality_files
    f = freq_files[0]
    df1 = pd.read_csv(f)
    cols = list(df1.columns)
    main_df = pd.DataFrame(columns=cols)
    for freq_file in freq_files:
        df = pd.read_csv(freq_file)
        main_df = pd.concat([main_df, df], ignore_index=True)
    # sorts dataframe output
    main_df.sort_values(['AAPos','Patient','Visit'],  inplace=True)
    # requires 100x read coverage
    main_df = main_df[main_df['Coverage'] >= 100]
    # drop positions > some position number
    main_df = drop_positions(main_df, position)
    return main_df