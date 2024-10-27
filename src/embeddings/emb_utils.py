import pandas as pd 

def get_lang_clusters(lang):
    data_dir = 'data/concreteness'
    translations_fname = 'conc_translated.csv'
    clusters_fname = 'Target Word Clusters.csv'

    translations = pd.read_csv(f'{data_dir}/{translations_fname}')
    clusters = pd.read_csv(f'{data_dir}/{clusters_fname}')

    translations = translations[translations['lang'] == lang]
    translations = translations.merge(clusters, on='word')
    return translations

if __name__ == '__main__':
    lang = 'de'
    data = get_lang_clusters(lang)
    print(data.head())