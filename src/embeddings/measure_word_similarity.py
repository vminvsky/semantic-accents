from tqdm import tqdm
from gensim.models import FastText
import json
import pandas as pd 
import numpy as np
import os 

def limited_most_similar(model, target_word, specific_words, top_n=100):
    """
    Computes the most similar words to the target word, limited to a specific list of words.

    Args:
        model: Gensim FastText model.
        target_word (str): The word for which to find similar words.
        specific_words (list): List of words to limit the search.
        top_n (int): Number of top similar words to return.

    Returns:
        List of tuples in the format [(word1, similarity1), (word2, similarity2), ...].
    """
    # Check if the target word is in the model's vocabulary
    if target_word not in model.wv:
        raise ValueError(f"Target word '{target_word}' not in the model's vocabulary.")

    # Filter out words not in the model's vocabulary
    filtered_words = [word for word in specific_words if word in model.wv]

    if not filtered_words:
        raise ValueError("None of the specific words are in the model's vocabulary.")

    # Get vectors for the filtered words
    word_vectors = np.array([model.wv[word] for word in filtered_words])

    # Get the vector for the target word
    target_vector = model.wv[target_word]

    # Compute cosine similarities between the target word and the filtered words
    similarities = model.wv.cosine_similarities(target_vector, word_vectors)

    # Remove NaN similarities
    valid_indices = ~np.isnan(similarities)
    if not np.any(valid_indices):
        return []  # Return an empty list if all similarities are NaN

    similarities = similarities[valid_indices]
    filtered_words = [word for idx, word in enumerate(filtered_words) if valid_indices[idx]]

    # Pair each word with its similarity score
    word_similarity_pairs = list(zip(filtered_words, similarities))

    # Sort the pairs by similarity score in descending order
    word_similarity_pairs.sort(key=lambda x: x[1], reverse=True)

    # Get the top N most similar words
    most_similar_words = word_similarity_pairs[:top_n]

    return most_similar_words

model_path_mapping = {
    'synthetic': '/scratch/gpfs/vv7118/projects/semantic-accents/embeddings/synthetic/{model_name}_{lang}{localized}/fasttext.model',
    'real': '/scratch/gpfs/vv7118/projects/semantic-accents/embeddings/real/human_{lang}/fasttext.model'
}

def main():
    langs = ['en', 'fr', 'et', 'hi', 'ja', 'ru', 'de']
    # langs = ['ru']
    settings = ['synthetic', 'real']
    prompt_formats = ['']#  ['', '_localized'] TODO LATER CHANGE TO ADD localized suffix
    word_source = 'concreteness' # 'concreteness' or 'NorthErualex_concepts'
    model_names = ['Meta-Llama-3-1-70B-Instruct-htzs', 'Meta-Llama-3-1-8B-Instruct-nwxcg']

    for prompt_format in prompt_formats:
        for model_name in model_names:
            for i, setting1 in enumerate(settings):
                for setting2 in settings[i:]:  # Only consider settings in the upper triangle, including the diagonal
                    for lang1 in tqdm(langs, desc="Processing languages for setting: " + setting1):
                        for lang2 in langs:  # Only consider languages after the current one
                            print(lang1, lang2)
                            data_rows = []
                            if (lang1 == lang2) & (setting1 == setting2):
                                continue
                            
                            src_emb = model_path_mapping[setting1].format(lang=lang1, model_name=model_name, localized=prompt_format)
                            tgt_emb = model_path_mapping[setting2].format(lang=lang2, model_name=model_name, localized=prompt_format)
                            try: 
                                m1 = FastText.load(src_emb)
                                m2 = FastText.load(tgt_emb)

                                mapping_path = f'data/{word_source}/mappings/{lang1}_{lang2}.json'
                                with open(mapping_path, 'r') as f:
                                    mapping = json.load(f)

                                # collect all target words
                                src_words = []
                                for word, _ in mapping.items():
                                    src_words.append(word)

                                # get the embeddings for all words
                                for word in tqdm(src_words, desc='looping over src words for langs - ' + lang1 + ' ' + lang2):
                                    tr_words = mapping[word]
                                    most_sim = limited_most_similar(m1, word, src_words, top_n=100)
                                    for sim_word, sim_score in most_sim:
                                        tr_sim_words = mapping[sim_word]
                                        for tr_word in tr_words:
                                            for tr_sim_word in tr_sim_words:
                                                sim = m2.wv.similarity(tr_word, tr_sim_word)
                                                data_rows.append({'src_word': word, 'tr_src_word': tr_word, 'tar_word': sim_word, 'tr_tar_word': tr_sim_word, 'sim': sim_score,'tr_sim': sim})
                                df = pd.DataFrame(data_rows)   
                                os.makedirs(f'data/{word_source}/similarity/{model_name}/{setting1}_{setting2}', exist_ok=True)
                                df.to_csv(f'data/{word_source}/similarity/{model_name}/{setting1}_{setting2}/{lang1}_{lang2}.csv', index=False)
                            except Exception as e:
                                print("didnt locate the model for ", lang1, lang2)
                                print(e)
                        

if __name__ == "__main__":
    main()
