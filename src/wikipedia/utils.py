# convert the json into correct prompt format 
import glob 
import json 
import os 
import pandas as pd 

from get_fresh_wiki_page import langs

list_words = ['List_of', 'Liste_der', 'Liste_']

def load_lang_wikipedia_data(lang, path='data/wikipedia/sections/json/{lang}/', path_specific=None):
    # load the lang data
    if path_specific is None:
        files_paths = glob.glob(os.path.join(path.format(lang=lang), '*.json'))
    else:
        files_paths = [path_specific]

    files_paths = [file_path for file_path in files_paths if not any(word in file_path for word in list_words)]

    lang_data = []
    for file_path in files_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            lang_data.append({
                'title': data['title'],
                'url': data['url'],
                'summary': data['summary'],
                'content': process_content(data['content']),
                'lang': lang
            })
    return lang_data

def process_content(content, depth=1):
    section_string = ""
    for section in content: 
        section_string += depth*'#' + section['section_title'] + '\n'
        if len(section['subsections']) > 0:
            section_string += process_content(section['subsections'], 
                                                 depth=depth+1)

    return section_string

def main():
    # sample_path = '/scratch/gpfs/vv7118/projects/semantic-accents/data/wikipedia/sections/json/en/Legalism_(Chinese_philosophy).json'
    langs = ['en', 'de', 'et', 'fr','hi','ja', 'ru'] # TODO: when finished, overwrite with others 
    os.makedirs('data/wikipedia/sections/processed/' ,exist_ok=True)
    for lang in langs: 
        lang_data = load_lang_wikipedia_data(lang)
        pd.DataFrame(lang_data).to_json(f'data/wikipedia/sections/processed/{lang}.jsonl', lines=True, orient='records', force_ascii=False)

if __name__ == '__main__':
    main()