import argparse
import json
import os
import pathlib
import re

import pandas as pd
import requests
import wikipediaapi
from bs4 import BeautifulSoup
from flair.data import Sentence
from flair.nn import Classifier
from tqdm import tqdm

from get_fresh_wiki_page import langs

def get_references(sentence, reference_dict):
    """
    Given a sentence, extract all reference index and find links from dictionary,
    then remove reference brackets from original sentences

    @param sentence, sentence to process
    @param reference_dict, dictionary of references
    @return cleaned sentence, reference_list pair
    """
    refs = re.findall(r'\[\d+\]', sentence)
    sentence = re.sub(r'\[\d+\]', '', sentence).strip().replace("\n", "")
    return sentence, [reference_dict[ref.replace("[", "").replace("]", "")] for ref in refs]


def extract_data(url, reference_dict):
    """
    Extract section data from wiki url.

    @param url: wiki url
    @reference_dict, reference dict from extract_references()
    @return a dictionary, key is section / subsection name, value is a list of {"sentence": ..., "refs": []}
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    data = {}
    for header in soup.find_all(['h1', 'h2', 'h3', "h4", "h5", "h6"]):
        section_title = header.text.replace('[edit]', '').strip().replace('\xa0', ' ')
        section_data = []
        for sibling in header.find_next_siblings():
            if sibling.name in ['h1', 'h2', 'h3', "h4", "h5", "h6"]:
                break
            if sibling.name == 'p':
                for sentence in sibling.text.replace("[", " [").split('. '):
                    if sentence:
                        sentence, refs = get_references(sentence, reference_dict)
                        if sentence:
                            section_data.append({"sentence": sentence, "refs": refs})
        data[section_title] = section_data
    return data


def extract_references(url):
    """
    Extract references from reference section with following structure:
    {
      "1": "https://...",
      "2": "https://...",
    }

    @param url, url of the wikipedia page
    @return dictionary of references, key is the citation index string, value is correpsonding url link
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    references = {}
    for references_section in soup.find_all('ol', {'class': 'references'}):
        for ref in references_section.find_all('li', {'id': lambda x: x and x.startswith('cite_note-')}):
            index = str(ref['id'].rsplit('-', 1)[-1])
            link_tag = ref.find('a', {'href': True, 'rel': 'nofollow'})
            isbn_tag = ref.find('a', href=re.compile(r'Special:BookSources'))
            if link_tag:
                link = link_tag['href']
                references[index] = link
            elif isbn_tag:
                link = f'ISBN: {isbn_tag.text}'
                references[index] = link
            else:
                references[index] = "[ERROR retrieving ref link]"
    return references


def getSections(page, structured_data):
    """
    Recursively extract each section title and plain text.

    @param page, page variable from wikipediaapi (e.g. wiki_api.page("page name"))
    @return a list of nested json for each section and corresponding subsections
    {
        "section_title": ...,
        "section_text": ...,
        "subsections": [
            {...},
            {...}
        ]
    }
    """
    return [{"section_title": i.title,
             "section_content": structured_data[i.title],
             "subsections": getSections(i, structured_data)
             } for i in page.sections]


def get_wikipedia_json_output(username, url, lang):
    """
    Get wikepdia output as format json

    @param username, username for wikipedia api agent.
    @param url, url of wikipedia page
    """
    wiki_api = wikipediaapi.Wikipedia(username, lang)
    wikipedia_page_name = url.replace(f"https://{lang}.wikipedia.org/wiki/", "")
    wikiapi_page = wiki_api.page(wikipedia_page_name)

    # extract references
    reference_dict = extract_references(url)
    structured_data = extract_data(url, reference_dict)

    # save extracted result to file
    result = {"title": wikipedia_page_name,
              "url": url,
              "summary": wikiapi_page.summary,
              "content": getSections(wikiapi_page, structured_data),
              "references": reference_dict}

    return result, wikipedia_page_name, reference_dict


def section_dict_to_text(data, inv_reference_dict, level=1):
    title = data["section_title"]
    content = data["section_content"]
    subsections = data["subsections"]
    if len(content) == 0 and len(subsections) == 0:
        return ""
    result = f"\n\n{'#' * level} {title}"
    if content:
        result += "\n\n"
        for cur_sentence in content:
            result += cur_sentence["sentence"]
            if cur_sentence["refs"]:
                result += " "
                result += " ".join(f"[{inv_reference_dict[ref]}]" for ref in cur_sentence["refs"] if
                                   ref != "[ERROR retrieving ref link]")
            result += ". "
    for subsection in subsections:
        result += section_dict_to_text(subsection, inv_reference_dict, level=level + 1)
    return result


def output_as_text(result, reference_dict):
    inv_reference_dict = {v: k for k, v in reference_dict.items()}
    output = result["title"] + "\n\n"
    output += result["summary"]
    for section in result["content"]:
        output += section_dict_to_text(section, inv_reference_dict)
    output += "\n\n# References\n\n"
    for idx, link in reference_dict.items():
        output += f"[{idx}] {link}\n"
    return output

# tagger = Classifier.load('ner')


# def extract_entities_flair(text):
#     sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)    
#     entities = []
#     for sentence in sentences:
#         if len(sentence) == 0:
#             continue
#         sentence = Sentence(sentence)
#         tagger.predict(sentence)
#         entities.extend([entity.text for entity in sentence.get_spans('ner')])

#     entities = list(set([e.lower() for e in entities]))

#     return entities

def process_url(url, output_dir, lang, username='Knowledge Curation Project'):
    wikipedia_page_name = url.replace(f"https://{lang}.wikipedia.org/wiki/", "")
    json_file_path = os.path.join(output_dir, 'json', lang, wikipedia_page_name + ".json")
    txt_file_path = os.path.join(output_dir, 'txt', lang, wikipedia_page_name + ".txt")

    if os.path.exists(json_file_path):
        print(f"File {json_file_path} already exists. Skipping extraction.")
        return

    result, wikipedia_page_name, reference_dict = get_wikipedia_json_output(username=username, url=url, lang=lang)
    txt = output_as_text(result, reference_dict)
    # clean_txt = re.sub(r'#+ ', '', re.sub(r'\[\d+\]', '', txt[:txt.find("\n\n# References\n\n")]))
    # Extract entities for future analysis.
    # result['flair_entities'] = extract_entities_flair(clean_txt)

    with open(json_file_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(txt_file_path, "w") as f:
        f.write(txt)

def main(output_dir, file_path, lang):
    pathlib.Path(f'{output_dir}/json/{lang}').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{output_dir}/txt/{lang}').mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(file_path)
    for _, row in tqdm(df.iterrows()):
        try:
            process_url(row['url'], output_dir, lang)
        except Exception as e:
            print(e)
            print(f'Error occurs when processing {row["url"]}')


if __name__ == '__main__':
    output_path = 'data/wikipedia/sections/'

    for lang in langs:
        file_path = f'data/wikipedia/{lang}_recent_edit_topic.csv'
        main(output_path, file_path, lang)