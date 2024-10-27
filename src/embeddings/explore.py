import pandas as pd
import numpy as np
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys
sys.path.append('src/')

from config import langs

def load_wordset() -> pd.DataFrame:
    """Load the wordset from the given path"""
    wordset_url = "data/concreteness/conc_translated.csv"
    df = pd.read_csv(wordset_url)
    return df

def load_fasttext_model(lang_code: str) -> FastText:
    """Load the FastText model for the given language code"""
    model_path = f"embeddings/Meta-Llama-3-1-70B-Instruct-htzs_{lang_code}/fasttext.model"
    model = FastText.load(model_path)
    return model

def load_bigram_model(lang_code: str):
    """Load the bigram model for the given language code, if needed"""
    bigram_path = f"embeddings/Meta-Llama-3-1-70B-Instruct-htzs_{lang_code}/bigram.pkl"
    # Check if the bigram model exists
    try:
        with open(bigram_path, 'rb') as f:
            bigram_model = pickle.load(f)
        return bigram_model
    except FileNotFoundError:
        return None

def main():
    # Load the wordset
    wordset_full = load_wordset()

    # Load the FastText models and bigram models for the specified languages
    models = {}
    bigram_models = {}
    for lang in langs:
        models[lang] = load_fasttext_model(lang)
        bigram_models[lang] = load_bigram_model(lang)

    # Collect embeddings for each word in the 'translation' column
    embeddings = []
    words = []
    for lang in langs:
        model = models.get(lang)
        bigram_model = bigram_models.get(lang)
        wordset = wordset_full[wordset_full['lang'] == lang]

        for _, row in wordset.iterrows():
            lang = row['lang']
            word = str(row['translation']).strip()

            if model:
                # If bigram model exists, apply it to the word
                if bigram_model:
                    word_tokens = word.split()
                    word = ' '.join(bigram_model[word_tokens])
                try:
                    # Get the embedding (FastText can handle OOV words)
                    embedding = model.wv[word]
                    embeddings.append(embedding)
                    words.append(word)
                except KeyError:
                    print(f"Could not generate embedding for word '{word}' in language '{lang}'.")
            else:
                print(f"No model found for language '{lang}'.")

            if not embeddings:
                print("No embeddings were found for the provided words.")
                return

            # Compute cosine similarity matrix
            embeddings_array = np.vstack(embeddings)
            similarity_matrix = cosine_similarity(embeddings_array)

            # Create a DataFrame with the similarity matrix
            similarity_df = pd.DataFrame(similarity_matrix, index=words, columns=words)
            # make similartiy df 16 bit
            similarity_df = similarity_df.astype(np.float8)

            # Optionally, save the similarity matrix to a CSV file
            similarity_df.to_csv(f"data/concreteness/{lang}_cosine_similarity_matrix.csv")


if __name__ == '__main__':
    similarity_df = main()
    print(similarity_df)