import gensim
from gensim.models import KeyedVectors
import numpy as np

from utils import printv, load_top_words

# Define concepts that may differ in color associations between English and French
concepts = {
    'en': ['apple', 'grass', 'sky', 'sun', 'blood', 'snow', 'night', 'banana', 'rose', 'water'],
    'fr': ['pomme', 'herbe', 'ciel', 'soleil', 'sang', 'neige', 'nuit', 'banane', 'rose', 'eau'],
    'de': ['apfel', 'gras', 'himmel', 'sonne', 'blut', 'schnee', 'nacht', 'banane', 'rose', 'wasser']
}

colors = {
        'en': [
            'red', 'green', 'blue', 'yellow', 'white', 'black'
        ],

        'fr': [
            'rouge', 'vert', 'bleu', 'jaune', 'blanc', 'noir'
        ],

        'de': [
            'rot', 'grün', 'blau', 'gelb', 'weiß', 'schwarz'
        ]
}

groundtruth = {
    'en': {
        'grass': 'green',
        'sky': 'blue',
        'sun': 'yellow',
        'blood': 'red',
        'snow': 'white',
        'banana': 'yellow',
        'rose': 'red',
        'rock': 'grey',
        'lime': 'green',
        'eggplant': 'purple',
        'ocean': 'blue',
        'tree': 'green',
        'cloud': 'white',
        'sand': 'beige',
        'coal': 'black',
        'apple': 'red',
        'carrot': 'orange',
        'grape': 'purple',
        'cherry': 'red',
        'lemon': 'yellow',
        'fire': 'orange',
        'butterfly': 'blue',
    },
    'fr': {
        'herbe': 'vert',
        'ciel': 'bleu',
        'soleil': 'jaune',
        'sang': 'rouge',
        'neige': 'blanc',
        'banane': 'jaune',
        'rose': 'rouge',
        'roche': 'gris',
        'citron': 'jaune',
        'aubergine': 'violet',
        'océan': 'bleu',
        'arbre': 'vert',
        'nuage': 'blanc',
        'sable': 'beige',
        'charbon': 'noir',
        'pomme': 'rouge',
        'carotte': 'orange',
        'raisin': 'violet',
        'cerise': 'rouge',
        'feu': 'orange',
        'papillon': 'bleu',
    },
    'de': {
        'gras': 'grün',
        'himmel': 'blau',
        'sonne': 'gelb',
        'blut': 'rot',
        'schnee': 'weiß',
        'banane': 'gelb',
        'rose': 'rot',
        'felsen': 'grau',
        'limette': 'grün',
        'aubergine': 'lila',
        'ozean': 'blau',
        'baum': 'grün',
        'wolke': 'weiß',
        'sand': 'beige',
        'kohle': 'schwarz',
        'apfel': 'rot',
        'karotte': 'orange',
        'traube': 'lila',
        'kirsche': 'rot',
        'feuer': 'orange',
        'schmetterling': 'blau',
    }
}

class Model: 
    def __init__(self, lang, path='data/embs/wiki.{}.vec', limit=20000):
        self.lang = lang
        self.path = path.format(lang)
        self.limit = limit
        self.model = KeyedVectors.load_word2vec_format(self.path,
                                                        binary=False, encoding='utf-8', 
                                                        limit=self.limit)
        
    def return_model(self):
        return self.model
    
    def shape(self):
        return self.model.vector_size

# Combine color vectors from both languages to compute the mean color vector
def get_embs(model, list_of_words: list):
    vectors = []
    valid_words = []
    for word in list_of_words:
        if word in model:
            vectors.append(model[word])
            valid_words.append(word)
        else:
            print(f"Word '{word}' not in vocabulary.")
    vectors = np.array(vectors)
    return vectors, valid_words

def create_color_dimension(model, concept_colors: dict):
    # v_diff = v_concept - v_color 
    # v_avg_diff = mean(v_diff)
    # return v_avg_diff
    color_vectors = []
    concept_vectors = []
    for concept, color in concept_colors.items():
        if concept in model and color in model:
            concept_vectors.append(model[concept])
            color_vectors.append(model[color])
        else:
            print(f"Concept '{concept}' or color '{color}' not in vocabulary.")
    concept_vectors = np.array(concept_vectors)
    color_vectors = np.array(color_vectors)
    color_dimensions = np.mean(color_vectors - concept_vectors, axis=0)
    return color_dimensions

def project_onto_colors(concept_vectors, color_dimensions):
    printv("concept_vectors.shape", concept_vectors.shape, verbose=1)
    printv("color_dimensions.shape", color_dimensions.shape, verbose=1)
    projections = np.dot(concept_vectors, color_dimensions.T) / \
        (np.linalg.norm(concept_vectors) * np.linalg.norm(color_dimensions))
    return projections

def return_similar_words(model):
    # Step 1: Extract all word vectors from the model
    words = list(model.key_to_index.keys())
    vectors = np.array([model[word] for word in words])

    # Step 2: Normalize the vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms

    # Step 3: Compute the cosine similarity matrix using dot product
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)

    # Step 4: Filter word pairs where the cosine similarity is greater than pi/3 (approx. 0.5)
    threshold = np.cos(np.pi / 3)

    # Get indices of the pairs that meet the similarity threshold (excluding diagonal)
    above_threshold = np.triu(similarity_matrix, k=1) > threshold

    # Step 5: Retrieve the word pairs with similarity greater than threshold
    similar_word_pairs = []

    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if above_threshold[i, j]:
                similar_word_pairs.append((words[i], words[j], similarity_matrix[i, j]))

    # Output the result
    return similar_word_pairs

def similar_words_compared_with_diff_vector(model, similar_word_pairs, diff_vector):
    diff_vector = diff_vector / np.linalg.norm(diff_vector)

    # Step 6: Compute cosine similarity between each word pair's difference vector and diff_vector
    word_pair_similarities = []
    for word1, word2, _ in similar_word_pairs:
        # Calculate the difference vector between the two words
        word1_vec = model[word1]
        word2_vec = model[word2]
        pair_diff_vector = word1_vec - word2_vec
        
        # Normalize the difference vector
        pair_diff_vector = pair_diff_vector / np.linalg.norm(pair_diff_vector)
        
        # Calculate cosine similarity with the target diff_vector
        similarity_with_diff_vector = np.dot(pair_diff_vector, diff_vector)
        
        # Store the word pair and similarity score
        word_pair_similarities.append((word1, word2, similarity_with_diff_vector))

    # Step 7: Sort word pairs by similarity with the diff_vector (highest to lowest)
    sorted_word_pairs = sorted(word_pair_similarities, key=lambda x: x[2], reverse=True)

    # Output the sorted list of word pairs
    return sorted_word_pairs

def measure_avg_color_association(concept_vecs, color_vecs, color_dim):
    """
    Noticing a problem where when we measure the sims there are a few colors
    that dominate it. 
    Idea: if we take the avg projections on each color, we can normalize by this
    """
    # get the embeddings for the top words
    concept_vecs = concept_vecs + color_dim
    # project the concept vectors onto the color dimensions
    projections = project_onto_colors(concept_vecs, color_vecs)
    # take the mean over the associations for each color
    avg_color_associations = np.mean(projections, axis=0)
    return avg_color_associations

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_top_colors(projections, concepts, colors, use_softmax=False):
    top_colors = []
    for i, concept in enumerate(concepts):
        concept_proj = projections[i]
        printv("concept_proj", concept_proj, verbose=1)
        if use_softmax:
            concept_proj = softmax(concept_proj)
        top_indices = np.argsort(concept_proj)[-3:][::-1]
        top_colors_scores = [(colors[idx], concept_proj[idx]) for idx in top_indices]
        top_colors.append((concept, top_colors_scores))
    return top_colors


def main():
    # Load the aligned embeddings for English and French
    # You need to download the aligned embeddings from the MUSE project:
    # https://github.com/facebookresearch/MUSE#download
    debug = True 
    verbose = 0
    if debug:
        verbose = 1

    use_color_dimensions = True

    limit = 20000 if debug else None
    lang1 = 'en'
    lang2 = 'de'
    print("Loading aligned English embeddings...")
    en_model = Model('en').return_model()
    print("Main embeddings loaded.")
    print('Loading similar words')
    en_similar = return_similar_words(en_model)

    print("Loading aligned other embeddings...")
    other_model = Model('de').return_model()
    print("Other embeddings loaded.")
    print('Loading similar words')
    other_similar = return_similar_words(other_model)

    colors1 = colors[lang1]
    colors2 = colors[lang2]

    concepts1 = concepts[lang1]
    concepts2 = concepts[lang2]

    color_dimensions1, valid_colors1 = get_embs(en_model, colors1)
    color_dimensions2, valid_colors2 = get_embs(other_model, colors2)

    concept_vectors1, concepts_filtered1 = get_embs(en_model, concepts1)
    concept_vectors2, concepts_filtered2 = get_embs(other_model, concepts2)

    c_abstract1 = create_color_dimension(en_model, groundtruth[lang1])
    c_abstract2 = create_color_dimension(other_model, groundtruth[lang2])

    sim_words1 = similar_words_compared_with_diff_vector(en_model, en_similar, c_abstract1)
    sim_words2 = similar_words_compared_with_diff_vector(other_model, other_similar, c_abstract2)

    print("Similar words for English:")
    for i, (word1, word2, sim) in enumerate(sim_words1):
        if i < 20:
            print(f"{word1} - {word2}: {sim}")

    print("Similar words for Other:")
    for i, (word1, word2, sim) in enumerate(sim_words2):
        if i < 20:
            print(f"{word1} - {word2}: {sim}")

    if use_color_dimensions:
        # we take the color - concept mean vec and add new concept.
        concept_vectors1 = concept_vectors1 + c_abstract1
        concept_vectors2 = concept_vectors2 + c_abstract2

    projections_en = project_onto_colors(concept_vectors1, color_dimensions1)
    projections_fr = project_onto_colors(concept_vectors2, color_dimensions2)

    printv("Projections EN:", projections_en, verbose=verbose)
    printv("Projections FR:", projections_fr, verbose=verbose)

    top_colors_en = get_top_colors(projections_en, concepts_filtered1, valid_colors1)
    top_colors_fr = get_top_colors(projections_fr, concepts_filtered2, valid_colors2)

    # Compare the colors associated with each concept in English and French
    print("Concept\tEnglish Color")
    for (concept_en, color_en) in top_colors_en:
        print(f"{concept_en}\t{color_en}")
    print("Concept\tOther Color")
    for (concept_en, color_en) in top_colors_fr:
        print(f"{concept_en}\t{color_en}")

if __name__ == '__main__':
    main()