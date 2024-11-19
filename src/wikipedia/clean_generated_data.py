# load in generated data 
# there are several key failure modes.
# - delete articles that are less than 100 chars
# 1.If a bullet point has less than 2 sentences drop. (DONE)
# 2. If the line starts with a table, drop. (DONE)
# 3. If line starts with a # drop. (DONE)
# ADJUST FOR REPETITIONS. (DONE, but kind of silly way)
# if a paragraph has less than 2 sentences drop. (DONE)
# 4. Remove new lines and convert them into one file. (DONE)
# - remove empty articles 
from pathlib import Path
from tqdm import tqdm 
import pandas as pd
import os 

import string
import unicodedata

import warnings
warnings.filterwarnings("ignore")

import regex as re

def remove_multilingual_punctuation(text):
    """
    Efficiently removes multilingual punctuation from the given text.

    Args:
        text (str): The input string.

    Returns:
        str: The string with punctuation removed.
    """
    # Define a regex pattern for Unicode punctuation (General Category: P)
    punctuation_regex = re.compile(r'[\p{P}]', re.UNICODE)
    
    # Substitute punctuation with an empty string
    return punctuation_regex.sub('', text)

def clean_data(df, column_name: str = 'article'):
    clean_data = []
    for i, row in enumerate(df[column_name]):
        if not isinstance(row, str):
            clean_data.append("")
            continue 
        if len(row) < 300: 
            clean_data.append("")
            continue
        lines = row.split('\n')
        data = []
        for line in lines: 
            stripped_line = str(line).strip() if line else ""
            if stripped_line is None:
                continue
            if not stripped_line or stripped_line.startswith("#"):
                continue
            if stripped_line.startswith("|"):
                continue 
            if stripped_line.startswith(("*", "-")):
                if len(stripped_line.split('.')) > 2:
                    data.append(stripped_line)
            else: 
                if len(str(stripped_line).split()) > 15:
                    if stripped_line in data: 
                        continue 
                    data.append(stripped_line)
        clean_data.append(" ".join(data))
    df['article'] = clean_data 
    df = df[df['article'].str.len() > 0]
    df['article'] = df['article'].str.lower()
    df['article'] = df['article'].apply(remove_multilingual_punctuation)
    
    return df


def load_data(data_source: str = 'synthetic'):
    if data_source == 'synthetic':
        base_dir = Path("data/wiki_gen/")
        output_dir = Path("data/wiki_gen_cleaned/")
    elif data_source == 'real':
        base_dir = Path("data/wikipedia/sections/extracted/")
        output_dir = Path("data/wikipedia/sections/cleaned/")

    for model_dir in base_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            
            # Iterate through language directories
            for lang_dir in tqdm(model_dir.iterdir()):
                print(lang_dir)
                if lang_dir.is_dir():
                    output_dir_temp = output_dir / model_name / lang_dir.name
                    os.makedirs(output_dir_temp, exist_ok=True)
                    language = lang_dir.name
                    
                    # Read all JSONL files in the directory
                    article_files_info = {}
                    for jsonl_file in lang_dir.glob("*.jsonl"):
                        modified_time = jsonl_file.stat().st_mtime
                        article_files_info[str(jsonl_file)] = modified_time
                        df = pd.read_json(jsonl_file, lines=True)
                        df = clean_data(df)
                        df.to_json(output_dir_temp / jsonl_file.name, orient='records', lines=True, force_ascii=False)

def main():
    load_data(data_source='synthetic')

if __name__ == "__main__":
    main()

