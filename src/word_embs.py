import gensim
from gensim.models import KeyedVectors
import numpy as np

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

def main():
    # Load the aligned embeddings for English and French
    # You need to download the aligned embeddings from the MUSE project:
    # https://github.com/facebookresearch/MUSE#download
    debug = True 

    limit = 20000 if debug else None
    lang1 = 'en'
    lang2 = 'de'
    print("Loading aligned English embeddings...")
    en_model = Model('en').return_model()
    print("Main embeddings loaded.")

    print("Loading aligned other embeddings...")
    other_model = Model('de').return_model()
    print("Other embeddings loaded.")

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
    colors1 = colors[lang1]
    colors2 = colors[lang2]


    # Combine color vectors from both languages to compute the mean color vector
    def get_combined_color_vectors(model, colors):
        vectors = []
        valid_colors = []
        for color in colors:
            if color in model:
                vectors.append(model[color])
                valid_colors.append(color)
            else:
                print(f"Color '{color}' not in vocabulary.")
        vectors = np.array(vectors)
        # mean_vector = np.mean(vectors, axis=0)
        # vectors = vectors - mean_vector
        return vectors, valid_colors

    color_dimensions1, valid_colors1 = get_combined_color_vectors(en_model, colors1)
    color_dimensions2, valid_colors2 = get_combined_color_vectors(other_model, colors2)

    # Define concepts that may differ in color associations between English and French
    concepts = {
        'en': ['apple', 'grass', 'sky', 'sun', 'blood', 'snow', 'night', 'banana', 'rose', 'water'],
        'fr': ['pomme', 'herbe', 'ciel', 'soleil', 'sang', 'neige', 'nuit', 'banane', 'rose', 'eau'],
        'de': ['apfel', 'gras', 'himmel', 'sonne', 'blut', 'schnee', 'nacht', 'banane', 'rose', 'wasser']
    }
    concepts1 = concepts[lang1]
    concepts2 = concepts[lang2]

    # Get vectors for concepts
    def get_concept_vectors(model, concepts):
        vectors = []
        valid_concepts = []
        for concept in concepts:
            if concept in model:
                vectors.append(model[concept])
                valid_concepts.append(concept)
            else:
                print(f"Concept '{concept}' not in vocabulary.")
        vectors = np.array(vectors)
        return vectors, valid_concepts

    concept_vectors1, concepts_filtered1 = get_concept_vectors(en_model, concepts1)
    concept_vectors2, concepts_filtered2 = get_concept_vectors(other_model, concepts2)

    # Project concepts onto color dimensions
    def project_onto_colors(concept_vectors, color_dimensions):
        # measure the cosine similarity
        projections = np.dot(concept_vectors, color_dimensions.T) / (np.linalg.norm(concept_vectors) * np.linalg.norm(color_dimensions))
        # projections = np.dot(concept_vectors, color_dimensions.T)
        return projections

    projections_en = project_onto_colors(concept_vectors1, color_dimensions1)
    projections_fr = project_onto_colors(concept_vectors2, color_dimensions2)

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def get_top_colors(projections, concepts, colors):
        top_colors = []
        for i, concept in enumerate(concepts):
            concept_proj = projections[i]
            softmax_scores = softmax(concept_proj)
            # assert softmax_scores.sum() == 1.0, f"Softmax scores do not sum to 1: {softmax_scores.sum()}"
            top_indices = np.argsort(concept_proj)[-3:][::-1]
            top_colors_scores = [(colors[idx], concept_proj[idx]) for idx in top_indices]
            top_colors.append((concept, top_colors_scores))
        return top_colors

    top_colors_en = get_top_colors(projections_en, concepts_filtered1, valid_colors1)
    top_colors_fr = get_top_colors(projections_fr, concepts_filtered2, valid_colors2)

    # Compare the colors associated with each concept in English and French
    print("Concept\tEnglish Color\tFrench Color")
    for (concept_en, color_en), (concept_fr, color_fr) in zip(top_colors_en, top_colors_fr):
        print(f"{concept_en}\t{color_en}\t{color_fr}")

if __name__ == '__main__':
    main()