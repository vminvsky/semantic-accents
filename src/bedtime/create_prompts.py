import pandas as pd 
import os 
from tqdm import tqdm
from collections import defaultdict

def load_conc_data(data_path: str = 'data/concreteness/conc_translated.csv'):
    data = pd.read_csv(data_path)
    return data

def process_lang(df, lang):
    lang_data = df[df['lang'] == lang]
    lang_data = lang_data[['word', 'translation']]
    lang_data.columns = ['en_src_word', '{}_word'.format(lang)]
    return lang_data

def convert_to_correct_format(data):
    langs = ['de','bn','ru']
    
    lang1_data = process_lang(data, 'en')

    for lang2 in tqdm(langs):
        lang2_data = process_lang(data, lang2)

        lang1_data = lang1_data.merge(
            lang2_data, on='en_src_word', how='inner')

    lang1_data.to_csv('data/concreteness/aligned_words.csv', index=False, encoding='utf-8')

def create_prompts(n_words: int = 10, num_samples=10000):
    data = pd.read_csv('data/concreteness/aligned_words.csv')
    langs = ['de','bn','ru', 'en']
    col_format = '{lang}_word'
    lang_list = defaultdict(list)
    for _ in tqdm(range(num_samples)):
        data = data.sample(n_words)
        for lang in langs:
            col = col_format.format(lang=lang)
            # convert to list of words
            words = data[col].tolist()
            lang_list[lang].append(words)

    output_dir = 'data/concreteness/random_words'
    os.makedirs(output_dir, exist_ok=True)

    for lang in langs:
        temp_df = pd.DataFrame(lang_list[lang]).apply(list, axis=1)
        temp_df.to_json(os.path.join(output_dir, f'{lang}.json'), orient='records', force_ascii=False, indent=4)


if __name__ == '__main__':
    data = load_conc_data()
    convert_to_correct_format(data)
    create_prompts()