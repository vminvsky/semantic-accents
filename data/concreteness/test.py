# Molly's data has a weird thing where the concreteness_tile and translation columns are sometimes switched.
# This function fixes that.
import pandas as pd 
import os 

data_dir = 'data/concreteness/'
fname = 'conc_translated.csv'

df = pd.read_csv(os.path.join(data_dir, fname))

print(df.head(30))