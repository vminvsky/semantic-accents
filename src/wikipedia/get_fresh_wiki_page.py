import logging
import re

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s : %(message)s")

langs = ['bn', 'en', 'de', 'ru']  # Add other language codes as needed
# langs = ['bn']

LANGS = {
    'en': "US",
    'de': "DE",
    'fr': "FR",
    'ru': "RU",
    'es': "ES",
    'ja': "JP",
    'bn': "BD",
}


def get_most_edited_wikipedia_titles(year: str, month: str, lang: str, day: str = "all-days", metric: str = 'pageviews'):
    headers = {
        "accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    if metric == 'pageviews':
        # a = requests.get(
        #     f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top-per-country/{LANGS[lang]}/all-access/{year}/{month}/{day}",
        #     headers=headers)
        a = requests.get(
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{lang}.wikipedia.org/all-access/{year}/{month}/{day}",
            headers=headers
        )
        results = a.json()['items'][0]['articles']
        titles = [result['article'].replace("_", " ") for result in results]
    else:
        a = requests.get(
            f"https://wikimedia.org/api/rest_v1/metrics/edited-pages/top-by-edits/{lang}.wikipedia/all-editor-types/content/{year}/{month}/{day}"
        )
        results = a.json()["items"][0]["results"][0]["top"]
        titles = [result["page_title"].replace("_", " ") for result in results]
    return titles


def get_html_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None


def call_ORES_api(revids, lang: str):
    """
    Example: https://en.wikipedia.org/w/index.php?title=Zagreb_Film&oldid=1182715485
    Put that into ORES > https://ores.wikimedia.org/v3/scores/enwiki?models=articlequality&revids=1182715485
    Return format:
    {"enwiki": {"models": {"articlequality": {"version": "0.9.2"}},
    "scores": {
      "1182715485": {
        "articlequality": {
          "score": {
            "prediction": "C",
            "probability": {"B": 0.13445393715303733, "C": 0.4728322988805659, "FA": 0.004610104723503904,
            "GA": 0.048191603091353855, "Start": 0.3326359821017828, "Stub": 0.007276074049756365}
          }
        }
      }}}}
    """
    base_url = f"https://ores.wikimedia.org/v3/scores/{lang}wiki"
    params = {
        "models": "articlequality",
        "revids": revids
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()[f'{lang}wiki']['scores'][f'{revids}']['articlequality']['score']
    except requests.RequestException as e:
        return None


def get_predicted_quality(title, lang):
    url = f'https://{lang}.wikipedia.org/w/index.php?title={title.replace(" ", "_")}&action=history'
    html_content = get_html_content(url)
    if html_content is None:
        logger.error(f'Cannot get the content of {url}')
        return None
    match = re.search(r'"wgCurRevisionId":(\d+)', html_content)
    if match:
        revids = match.group(1)
    else:
        logger.error(f'Cannot find revids for {title}.')
        return None

    predicted_quality = call_ORES_api(revids, lang=lang)
    return predicted_quality


def main():
    # You can change the time_periods to get the most edited Wikipedia articles in different time periods.
    time_periods = [("2021", "01"), ("2021", "02"), ("2021", "03"), ("2021", "04"), ("2021", "05"), ("2021", "06"),
                    ("2021", "07"), ("2021", "08"), ("2021", "09"), ("2021", "10"), ("2021", "11"), ("2021", "12"),
                    ("2022", "01"),
                    ("2022", "02"), ("2022", "03"), ("2022", "04"), ("2022", "05"), ("2022", "06"), ("2022", "07"),
                    ("2022", "08"), ("2022", "09"), ("2022", "10"), ("2022", "11"), ("2022", "12"),
                    ("2023", "01"), ("2023", "02"), ("2023", "03"), ("2023", "04"), ("2023", "05"), ("2023", "06"),
                    ("2023", "07"), ("2023", "08"), ("2023", "09"), ('2024', '01'), ('2024', '02'), ('2024', '03'),
                    ('2024', '04'), ('2024', '05'), ('2024', '06'), ('2024', '07'), ('2024', '08'), ('2024', '09'),]


    for lang in langs:
        data = {
            'topic': [],
            'url': [],
            'date': [],
            # 'predicted_class': [],
            # 'predicted_scores': []
        }
        print(f"Processing {lang}...")
        for year, month in tqdm(time_periods):
            for metric in ['edits','pageviews']:
                titles = get_most_edited_wikipedia_titles(year, month, lang, metric=metric)
                date_format = f"{year}-{month}"
                for title in titles:
                    # predicted_quality = get_predicted_quality(title, lang)
                    # if predicted_quality is None:
                    #     logger.error(f'Fail to include "{title}"')
                    #     continue
                    data['topic'].append(title)
                    data['url'].append(f'https://{lang}.wikipedia.org/wiki/{title.replace(" ", "_")}')
                    data['date'].append(date_format)
                    # data['predicted_class'].append(predicted_quality['prediction'])
                    # data['predicted_scores'].append(predicted_quality['probability'])

        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=['topic'])
        df.to_csv(f'data/wikipedia/{lang}_recent_edit_topic.csv')


if __name__ == '__main__':
    main()