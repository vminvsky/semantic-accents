import json
import os
import pandas as pd 

langs = ['et','fr','hi','jp', 'de', 'en']


for lang1 in langs: 
    print(lang1)
    data = pd.read_csv(f'{lang1}.csv')[['concept_id', 'ortho_form']]
    data.columns = ['concept_id', lang1]
    for lang2 in langs:
        print(lang2)
        if lang1 == lang2:
            continue
        temp = pd.read_csv(f'{lang2}.csv')
        temp[lang2] = temp['ortho_form']
        data2 = data.merge(temp[['concept_id', lang2]], on='concept_id', how='inner')
        # create a dictionary mapping words from the two langs
        word_dict = {}
        data2 = data2.groupby(lang1)[lang2].apply(list).reset_index()
        for i, row in data2.iterrows():
            word_dict[row[lang1]] = row[lang2]
        
        # save the dictionary to a folder called mappings with name {lang1}_{lang2}.json
        os.makedirs('mappings', exist_ok=True)
        with open(f'mappings/{lang1}_{lang2}.json', 'w') as f:
            json.dump(word_dict, f, indent=4, ensure_ascii=False)
