import pandas as pd
import numpy as np
from gensim.models import FastText, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys
import os
import argparse
import logging
sys.path.append('src/')

from emb_utils import get_lang_clusters
from config import langs

langs = ['de', 'en']

def load_model(lang_code: str, use_custom_models: bool, model_name: str =f"Meta-Llama-3-1-70B-Instruct-htzs"):
    """Load the model for the given language code."""
    if use_custom_models:
        model_path = f"embeddings/{model_name}_{lang_code}/fasttext.model"
        try:
            logging.info(f"Loading custom model for language '{lang_code}' from '{model_path}'.")
            model = FastText.load(model_path)
            return model
        except Exception as e:
            logging.error(f"Error loading custom model for '{lang_code}': {e}")
            return None
    else:
        model_path = f"data/embs/wiki.{lang_code}.vec"
        try:
            logging.info(f"Loading pretrained embeddings for language '{lang_code}' from '{model_path}'.")
            model = KeyedVectors.load_word2vec_format(model_path, binary=False, limit=100000)
            return model
        except Exception as e:
            logging.error(f"Error loading pretrained embeddings for '{lang_code}': {e}")
            return None

def load_bigram_model(lang_code: str, model_name: str):
    """Load the bigram model for the given language code, if needed."""
    bigram_path = f"embeddings/{model_name}_{lang_code}/bigram.pkl"
    try:
        logging.info(f"Loading bigram model for language '{lang_code}' from '{bigram_path}'.")
        with open(bigram_path, 'rb') as f:
            bigram_model = pickle.load(f)
        return bigram_model
    except FileNotFoundError:
        logging.warning(f"Bigram model not found for language '{lang_code}' at '{bigram_path}'.")
        return None
    except Exception as e:
        logging.error(f"Error loading bigram model for '{lang_code}': {e}")
        return None

def main(use_custom_models):
    model_name = "Meta-Llama-3-1-70B-Instruct-htzs"
    prefix = 'custom' if use_custom_models else 'pretrained'
    logging.info(f"Starting processing with '{prefix}' models.")

    for lang in langs:
        logging.info(f"Processing language: {lang}")

        # Load the model
        model = load_model(lang, use_custom_models, model_name=model_name)
        if model is None:
            logging.warning(f"Skipping language '{lang}' due to model loading issues.")
            continue

        bigram_model = load_bigram_model(lang, model_name=model_name) if use_custom_models else None

        data = get_lang_clusters(lang)
        if data is None or data.empty:
            logging.warning(f"No data for language '{lang}'. Skipping.")
            continue

        data['embedding'] = None 

        missing_embeddings = 0
        total_words = len(data)
        for idx, row in data.iterrows():
            word = str(row['translation']).strip()

            if bigram_model:
                word_tokens = word.split()
                word = ' '.join(bigram_model[word_tokens])

            try:
                if use_custom_models:
                    # For FastText models
                    embedding = model.wv[word]
                else:
                    # For KeyedVectors
                    embedding = model[word]
                data.at[idx, 'embedding'] = embedding  # Store embedding in data
            except KeyError:
                logging.debug(f"Word '{word}' not found in embeddings for language '{lang}'.")
                missing_embeddings += 1

        logging.info(f"Total words: {total_words}, Missing embeddings: {missing_embeddings}")

        # Remove rows where embedding was not found
        data = data.dropna(subset=['embedding']).reset_index(drop=True)

        if data.empty:
            logging.warning(f"No embeddings generated for language '{lang}'. Skipping.")
            continue

        # Convert embeddings to numpy arrays
        data['embedding'] = data['embedding'].apply(lambda x: np.array(x))

        # Group data by cluster
        cluster_groups = data.groupby('cluster')

        intra_cluster_distances = {}

        for cluster, group in cluster_groups:
            embeddings = np.stack(group['embedding'].values)
            similarity_matrix = cosine_similarity(embeddings)
            distance_matrix = 1 - similarity_matrix

            # Get the upper triangle of the distance matrix, excluding the diagonal
            triu_indices = np.triu_indices_from(distance_matrix, k=1)
            distances = distance_matrix[triu_indices]
            if len(distances) > 0:
                avg_distance = np.mean(distances)
                intra_cluster_distances[cluster] = avg_distance
                logging.debug(f"Intra-cluster average distance for cluster '{cluster}': {avg_distance}")
            else:
                intra_cluster_distances[cluster] = None  # Only one element in cluster
                logging.debug(f"Cluster '{cluster}' has only one element. Intra-cluster distance is set to None.")

        # Compute cluster centroids
        cluster_centroids = {}
        for cluster, group in cluster_groups:
            embeddings = np.stack(group['embedding'].values)
            centroid = embeddings.mean(axis=0)
            cluster_centroids[cluster] = centroid
            logging.debug(f"Computed centroid for cluster '{cluster}'.")

        # Compute inter-cluster distances between centroids
        clusters = list(cluster_centroids.keys())
        inter_cluster_distances = {}
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_i = clusters[i]
                cluster_j = clusters[j]
                centroid_i = cluster_centroids[cluster_i]
                centroid_j = cluster_centroids[cluster_j]
                similarity = cosine_similarity([centroid_i], [centroid_j])[0][0]
                distance = 1 - similarity
                inter_cluster_distances[(cluster_i, cluster_j)] = distance
                logging.debug(f"Inter-cluster distance between '{cluster_i}' and '{cluster_j}': {distance}")

        # Create a directory for the language results
        result_dir = f'results/{model_name}/{lang}/{prefix}'
        os.makedirs(result_dir, exist_ok=True)
        logging.info(f"Saving results to '{result_dir}'.")

        intra_cluster_df = pd.DataFrame(
            list(intra_cluster_distances.items()), columns=['cluster', 'avg_intra_distance']
        )
        intra_cluster_df.to_csv(f'{result_dir}/intra_cluster_distances_{lang}.csv', index=False)

        inter_cluster_df = pd.DataFrame(
            [(c[0], c[1], d) for c, d in inter_cluster_distances.items()],
            columns=['cluster1', 'cluster2', 'inter_distance']
        )
        inter_cluster_df.to_csv(f'{result_dir}/inter_cluster_distances_{lang}.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute intra-cluster and inter-cluster distances.')
    parser.add_argument('--use_custom_models', action='store_true',
                        help='Use custom FastText models instead of pretrained embeddings.')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set the logging level.')
    parser.add_argument('--process_both', default=True, action='store_true',
                        help='Process both custom and pretrained models.')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if args.process_both:
        logging.info("Processing both custom and pretrained models.")
        main(use_custom_models=True)
        main(use_custom_models=False)
    else:
        main(use_custom_models=args.use_custom_models)