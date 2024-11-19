import os 
from dataclasses import dataclass 
import json 
from tqdm import tqdm
import glob

from get_fresh_wiki_page import langs

@dataclass 
class WikipediaData:
    title: str 
    article: str 
    lang: str 

def load_file(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def process_section(article_text: str, section: dict):
    article_text += '\n'+  section['section_title']

    sentences_in_section = []
    for sentence in section['section_content']:
        sentences_in_section.append(sentence['sentence'])

    article_text += '\n' + ' '.join(sentences_in_section) + '\n'
    if len(section['subsections']) > 0:
        # print(section['section_title'])
        for subsec in section['subsections']:
            article_text = process_section(article_text, subsec)
    return article_text 

def main(lang: str):
    data_path = f'data/wikipedia/sections/json/{lang}/'
    output_path = f'data/wikipedia/sections/extracted/human/{lang}/'

    os.makedirs(output_path, exist_ok=True)
    all_data = []
    for file in tqdm(glob.glob(f'{data_path}*.json'), desc=f'Processing {lang}'):
        article = load_file(file)
        article_text = ''
        content = article['content']
        for c in content:
            article_text = process_section(article_text, c)
        data = {'title': article['title'], 'article': article_text, 'lang': lang}
        all_data.append(data)

    output_file = f'{output_path}articles.jsonl'
    with open(output_file, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n', ensure_ascii=False)

if __name__ == '__main__':
    for lang in langs:
        main(lang)