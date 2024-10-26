import requests
import pandas as pd
from datetime import datetime
import os 

def get_top_articles(language, year, month, limit=10):
    """
    Fetch the top-viewed articles from Wikipedia for a given language, year, and month.

    Args:
        language (str): Language code (e.g., 'en', 'de', 'fr').
        year (int): Year to fetch data for.
        month (int): Month to fetch data for.
        limit (int): Number of top articles to retrieve.

    Returns:
        list: List of dictionaries containing article title and view count.
    """
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{language}.wikipedia/all-access/{year}/{month:02d}/all-days"
    response = requests.get(url)
    
    if response.status_code == 200:
        articles = response.json().get('items', [])[0].get('articles', [])
        return articles[:limit]
    else:
        print(response.text)
        print(f"Failed to fetch data for {language}-{year}-{month}")
        return []

def collect_articles_for_year(languages, year, limit=10):
    """
    Collect top articles for each month in a specified year across multiple languages.

    Args:
        languages (list): List of language codes to fetch data for.
        year (int): Year to fetch data for.
        limit (int): Number of top articles to retrieve per month.

    Returns:
        DataFrame: Combined data for all languages, months, and top articles.
    """
    all_data = []
    
    for language in languages:
        for month in range(1, 13):
            articles = get_top_articles(language, year, month, limit)
            for article in articles:
                all_data.append({
                    'language': language,
                    'year': year,
                    'month': month,
                    'article': article['article'],
                    'views': article['views']
                })
    
    return pd.DataFrame(all_data)

def main():
    # Parameters
    languages = ['en', 'de', 'fr', 'es', 'ja']  # Add other language codes as needed
    year = 2022
    limit = 1000  # Number of top articles to retrieve each month
    save_dir = 'data/wikipedia/'

    # Collect data
    data = collect_articles_for_year(languages, year, limit)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"top_wikipedia_articles_{year}_{timestamp}.csv"
    file_path = os.path.join(save_dir, filename)  
    data.to_csv(file_path, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    main()