from tqdm import tqdm
from gensim.models import FastText, KeyedVectors
import json
import pandas as pd 
import numpy as np
import os 

model_path_mapping = {
    'synthetic': '/scratch/gpfs/vv7118/projects/semantic-accents/embeddings/synthetic/{model_name}_{lang}{localized}/fasttext.model',
    'real': '/scratch/gpfs/vv7118/projects/semantic-accents/embeddings/real/human_{lang}/fasttext.model',
    # 'original': '/scratch/gpfs/vv7118/projects/semantic-accents/embeddings/real/{lang}/fasttext.model',
    'original': '/scratch/gpfs/vv7118/projects/semantic-accents/data/embs/wiki.{lang}.vec',
    'synth-bedtime': '/scratch/gpfs/vv7118/projects/semantic-accents/embeddings/synth-bedtime/{model_name}_{lang}{localized}/fasttext.model'
}

# TODO: add a way to get a random baseline. 
# - for each word, select k random words instead of the most similar.
# - compare the similarity of the translations of the random words to the translations of the most similar words

def load_model(model_path):
    if 'wiki' in model_path:
        model =  KeyedVectors.load_word2vec_format(model_path, limit=80000)
    else:
        model = FastText.load(model_path).wv
    return model

def main():
    """
    Rough outline of what should happen
    1. get source words 
    2. measure similarities btwn those source words
    3. for each word in source words find k nearest words.
    4. for source word and k nearest translate them 
    5. measure the similartiy between the tr src word and tr tar word.
    6. loop over multiple possible translations for the words 
    """
    langs = ['de','en', 'bn', 'ru']
    settings = ['synthetic', 'original']
    prompt_formats = ['']  # TODO: Add localized suffix if needed
    word_source = 'concreteness'  # 'concreteness' or 'NorthErualex_concepts'
    model_names = ['Meta-Llama-3-1-8B-Instruct-nwxcg']
    mapping_name = 'mappings' # 'mappings', 'mappings_counts_geq_5'
    top_n = 25
    threshold = 0.5 # optional threshold 

    for prompt_format in prompt_formats:
        for model_name in model_names:
            for i, setting1 in enumerate(settings):
            # for i, setting1 in enumerate(tqdm(['synthetic'], desc="Processing settings")):
                for setting2 in settings[i:]:  # Upper triangle including diagonal
                    if setting2 == 'synth-bedtime':
                        continue
                # for setting2 in ['real']:
                    for lang1 in tqdm(langs, desc=f"Processing languages for setting: {setting1}"):
                        for lang2 in langs:
                            if (lang1 == lang2) and (setting1 == setting2):
                                continue
                            print(f"Processing language pair: {lang1}, {lang2}")
                            data_rows = []

                            src_emb_path = model_path_mapping[setting1].format(lang=lang1, model_name=model_name, localized=prompt_format)
                            tgt_emb_path = model_path_mapping[setting2].format(lang=lang2, model_name=model_name, localized=prompt_format)
                            try:
                                m1 = load_model(src_emb_path)
                                m2 = load_model(tgt_emb_path)

                                mapping_path = f'data/{word_source}/{mapping_name}/{lang1}_{lang2}.json'
                                with open(mapping_path, 'r') as f:
                                    mapping = json.load(f)

                                src_words = list(mapping.keys())

                                # Precompute word vectors for all src_words in m1
                                src_word_vectors = {word: m1[word] for word in src_words if word in m1}
                                src_word_list = list(src_word_vectors.keys())
                                src_vectors = np.array(list(src_word_vectors.values()))

                                # Normalize vectors for cosine similarity
                                src_norms = np.linalg.norm(src_vectors, axis=1, keepdims=True)
                                normalized_src_vectors = src_vectors / src_norms

                                # Compute similarity matrix
                                similarity_matrix = np.dot(normalized_src_vectors, normalized_src_vectors.T)

                                # For each word, get top N most similar words
                                for idx, word in enumerate(tqdm(src_word_list, desc=f'Processing words for {lang1}-{lang2}')):
                                    try: 
                                        tr_words = mapping[word]
                                        similarities = similarity_matrix[idx]
                                        similarities[idx] = -np.inf  # Exclude self-similarity

                                        # Get top N indices
                                        # Filter similarities above the threshold
                                        indices_above_threshold = np.where(similarities > threshold)[0]
                                        # print(len(indices_above_threshold))
                                        if len(indices_above_threshold) == 0:
                                            continue  # Skip if no similar words above threshold

                                        similarities_above_threshold = similarities[indices_above_threshold]
                                        num_to_get = min(top_n, len(similarities_above_threshold))
                                        sorted_indices = np.argsort(-similarities_above_threshold)[:num_to_get]
                                        top_indices = indices_above_threshold[sorted_indices]
                                        top_similarities = similarities[top_indices]
                                        top_words = [src_word_list[i] for i in top_indices]

                                        # top_indices = np.argpartition(-similarities, top_n)[:top_n]
                                        # top_similarities = similarities[top_indices]
                                        # top_words = [src_word_list[i] for i in top_indices]

                                        # Prepare translation word vectors
                                        tr_word_set = set(tr_words)
                                        tr_word_vectors = {w: m2[w] for w in tr_word_set if w in m2}

                                        # Process each similar word
                                        for sim_word, sim_score in zip(top_words, top_similarities):
                                            tr_sim_words = mapping.get(sim_word, [])
                                            tr_sim_word_set = set(tr_sim_words)
                                            tr_sim_word_vectors = {w: m2[w] for w in tr_sim_word_set if w in m2}

                                            # Skip if no translation vectors found
                                            if not tr_word_vectors or not tr_sim_word_vectors:
                                                continue

                                            # Prepare arrays for vectorized computation
                                            tr_word_vecs = np.array(list(tr_word_vectors.values()))
                                            tr_sim_word_vecs = np.array(list(tr_sim_word_vectors.values()))

                                            # Normalize vectors
                                            tr_word_norms = np.linalg.norm(tr_word_vecs, axis=1, keepdims=True)
                                            tr_sim_word_norms = np.linalg.norm(tr_sim_word_vecs, axis=1, keepdims=True)
                                            tr_word_vecs_norm = tr_word_vecs / tr_word_norms
                                            tr_sim_word_vecs_norm = tr_sim_word_vecs / tr_sim_word_norms

                                            # Compute cosine similarity matrix
                                            tr_similarity_matrix = np.dot(tr_word_vecs_norm, tr_sim_word_vecs_norm.T)

                                            # Collect data rows
                                            for i, tr_word in enumerate(tr_word_vectors.keys()):
                                                for j, tr_sim_word in enumerate(tr_sim_word_vectors.keys()):
                                                    tr_sim = tr_similarity_matrix[i, j]
                                                    data_rows.append({
                                                        'src_word': word,
                                                        'tr_src_word': tr_word,
                                                        'tar_word': sim_word,
                                                        'tr_tar_word': tr_sim_word,
                                                        'sim': sim_score,
                                                        'tr_sim': tr_sim
                                                    })
                                    except Exception as e:
                                        print(f"Error processing word {word}: {e}")
                                        continue

                                # Save results
                                output_dir = f'data/{word_source}/similarity/{model_name}/{setting1}_{setting2}'
                                os.makedirs(output_dir, exist_ok=True)
                                df = pd.DataFrame(data_rows)
                                df.to_csv(f'{output_dir}/{lang1}_{lang2}.csv', index=False)
                            except Exception as e:
                                print(f"Error processing {lang1}-{lang2}: {e}")

if __name__ == "__main__":
    main()