# Molly's data has a weird thing where the concreteness_tile and translation columns are sometimes switched.
# This function fixes that.
# NOTE IT MIGHT BE WORDS SPELLED BACKWORDS FOR SOME LANGS -- WEIRD
import pandas as pd 
import os 

data_dir = 'data/concreteness/'
fname = 'Target Translations XLing Words.csv'

# Function to swap if concreteness_tile isn't numeric
def fix_row(row):
    try:
        # Try converting concreteness_tile to an integer
        row['concreteness_tile'] = int(row['concreteness_tile'])
    except ValueError:
        # If conversion fails, it’s likely a word, so swap
        print(row['concreteness_tile'])
        row['translation'], row['concreteness_tile'] = row['concreteness_tile'], row['translation']
    return row

df = pd.read_csv(os.path.join(data_dir, fname))

# Apply the function to each row
df = df.apply(fix_row, axis=1)

# Save the fixed data
df.to_csv(os.path.join(data_dir, 'conc_translated.csv'), index=False)