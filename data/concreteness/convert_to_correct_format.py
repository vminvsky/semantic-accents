# james' data not in the same format as the northerualex_concepts
# so we need to convert it to the same format

# current format
# lang, en_word, tr_word, concreteness_tile 

# ideal_format
# mappings folder:
# for each lang1-lang2 pair have a dict with the following format
#    {src_word1: tr_word1, src_word2: tr_word2, ...}
import pandas as pd 
import os 
from tqdm import tqdm

def load_conc_data(data_path: str = 'data/concreteness/conc_translated.csv'):
    data = pd.read_csv(data_path)
    return data

def process_lang(df, lang):
    lang_data = df[df['lang'] == lang]
    lang_data = lang_data[['word', 'translation']]
    lang_data.columns = ['en_src_word', '{}_word'.format(lang)]
    return lang_data

def convert_to_correct_format(data):
    langs = data['lang'].unique()
    
    for lang1 in tqdm(langs):
        for lang2 in langs: 
            lang1_data = process_lang(data, lang1)
            lang2_data = process_lang(data, lang2)

            merged = lang1_data.merge(
                lang2_data, on='en_src_word', how='inner')
            merged.drop('en_src_word', axis=1, inplace=True)

            if lang1 == lang2: 
                merged = merged.groupby('{}_word_x'.format(lang1))['{}_word_y'.format(lang2)].apply(list)

            else: 
                merged = merged.groupby('{}_word'.format(lang1))['{}_word'.format(lang2)].apply(list)
            save_dir = 'data/concreteness/mappings'
            os.makedirs(save_dir, exist_ok=True)
            merged.to_json('data/concreteness/mappings/{}_{}.json'.format(lang1, lang2), indent=4, orient='index', force_ascii=False)

if __name__ == '__main__':
    data = load_conc_data()
    convert_to_correct_format(data)