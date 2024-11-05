import pandas as pd 
from dataclasses import dataclass 

from utils import ctry_lang

@dataclass
class Analysis:
    lang: str 
    country: str 
    no_country: bool 
    limit: bool 
    eval_model_name: str
    translate_model_name: str
    score: float


def extract_answer(logprobs):
    try: 
        if isinstance(logprobs, str):
            # return the first value of the dict
            logprobs = eval(logprobs)
        return list(logprobs.keys())[0]
    except Exception as e:
        print(f'error in extracting log propbs - {e}')
        print(logprobs)
        return None

def run_analysis(df, lang, limit_ctry=True):
    # if no country limit to rows where the country col is equal to lang
    if limit_ctry:
        df = df[df['country'] == ctry_lang[lang]]

    # calculate precision recall 
    df['correct'] = df['answer_gen'] == df['answer']
    df['correct'] = df['correct'].astype(int)
    return df 


def main():
    ctry_settings = [True, False]
    limit = True 
    langs = ['de', 'en', 'ja', 'ru']
    eval_model_name = 'gpt-35-turbo-16k'
    translate_model_name = 'gpt-4o'
    scores = []
    for no_country in ctry_settings: 
        for lang in langs: 
            print(lang)
            str_ctry = f'eval_results_no_country_{eval_model_name}.jsonl' if no_country else f'eval_results_with_country_{eval_model_name}.jsonl'
            data_path = f'data/cultural_bench/{translate_model_name}/{lang}/{str_ctry}'

            df = pd.read_json(data_path, lines=True)
            country = ctry_lang[lang]
            df['answer_gen'] = df['logprobs'].apply(extract_answer)
            df.dropna(subset='answer_gen', inplace=True)
            df = run_analysis(df, lang, limit)
            score = df.groupby('country')['correct'].mean().sort_values().values[0]
            scores.append(Analysis(lang, country, no_country, limit, eval_model_name, translate_model_name, score))
    
    df = pd.DataFrame(scores)
    df.to_csv(f'data/cultural_bench/{translate_model_name}/analysis_results_{eval_model_name}.csv', index=False)

if __name__ == "__main__":
    main()