from tqdm import tqdm
from gensim.models import FastText
import json
import pandas as pd 
import numpy as np
import os 

model_path_mapping = {
    'synthetic': '/scratch/gpfs/vv7118/projects/semantic-accents/embeddings/synthetic/{model_name}_{lang}{localized}/fasttext.model',
    'real': '/scratch/gpfs/vv7118/projects/semantic-accents/embeddings/real/human_{lang}/fasttext.model'
}

def main():
    langs = ['en', 'fr', 'et', 'hi', 'ja', 'ru', 'de']
    settings = ['synthetic', 'real']
    prompt_formats = ['']  # TODO: Add localized suffix if needed
    word_source = 'concreteness'  # 'concreteness' or 'NorthErualex_concepts'
    model_names = ['Meta-Llama-3-1-70B-Instruct-htzs', 'Meta-Llama-3-1-8B-Instruct-nwxcg']
    top_n = 100

    for prompt_format in prompt_formats:
        for model_name in model_names:
            for i, setting1 in enumerate(settings):
                for setting2 in settings[i:]:  # Upper triangle including diagonal
                    for lang1 in tqdm(langs, desc=f"Processing languages for setting: {setting1}"):
                        for lang2 in langs:
                            if (lang1 == lang2) and (setting1 == setting2):
                                continue
                            print(f"Processing language pair: {lang1}, {lang2}")
                            data_rows = []

                            src_emb_path = model_path_mapping[setting1].format(lang=lang1, model_name=model_name, localized=prompt_format)
                            tgt_emb_path = model_path_mapping[setting2].format(lang=lang2, model_name=model_name, localized=prompt_format)
                            try:
                                m1 = FastText.load(src_emb_path)
                                m2 = FastText.load(tgt_emb_path)

                                mapping_path = f'data/{word_source}/mappings/{lang1}_{lang2}.json'
                                with open(mapping_path, 'r') as f:
                                    mapping = json.load(f)

                                src_words = list(mapping.keys())

                                # Precompute word vectors for all src_words in m1
                                src_word_vectors = {word: m1.wv[word] for word in src_words if word in m1.wv}
                                src_word_list = list(src_word_vectors.keys())
                                src_vectors = np.array(list(src_word_vectors.values()))

                                # Normalize vectors for cosine similarity
                                src_norms = np.linalg.norm(src_vectors, axis=1, keepdims=True)
                                normalized_src_vectors = src_vectors / src_norms

                                # Compute similarity matrix
                                similarity_matrix = np.dot(normalized_src_vectors, normalized_src_vectors.T)

                                # For each word, get top N most similar words
                                for idx, word in enumerate(tqdm(src_word_list, desc=f'Processing words for {lang1}-{lang2}')):
                                    tr_words = mapping[word]
                                    similarities = similarity_matrix[idx]
                                    similarities[idx] = -np.inf  # Exclude self-similarity

                                    # Get top N indices
                                    top_indices = np.argpartition(-similarities, top_n)[:top_n]
                                    top_similarities = similarities[top_indices]
                                    top_words = [src_word_list[i] for i in top_indices]

                                    # Prepare translation word vectors
                                    tr_word_set = set(tr_words)
                                    tr_word_vectors = {w: m2.wv[w] for w in tr_word_set if w in m2.wv}

                                    # Process each similar word
                                    for sim_word, sim_score in zip(top_words, top_similarities):
                                        tr_sim_words = mapping.get(sim_word, [])
                                        tr_sim_word_set = set(tr_sim_words)
                                        tr_sim_word_vectors = {w: m2.wv[w] for w in tr_sim_word_set if w in m2.wv}

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

                                # Save results
                                output_dir = f'data/{word_source}/similarity/{model_name}/{setting1}_{setting2}'
                                os.makedirs(output_dir, exist_ok=True)
                                df = pd.DataFrame(data_rows)
                                df.to_csv(f'{output_dir}/{lang1}_{lang2}.csv', index=False)
                            except Exception as e:
                                print(f"Error processing {lang1}-{lang2}: {e}")

if __name__ == "__main__":
    main()