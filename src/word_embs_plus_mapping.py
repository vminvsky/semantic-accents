import gensim
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import joblib

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
    
def load_colors(langs = ['en','fr', 'de']):
    color_dict = {}
    for lang in langs: 
        df = pd.read_csv(f'data/{lang}.csv')
        color_dict[lang] = df['color'].to_list()
    return color_dict

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

    # Load the MLP models
    mlp_en = joblib.load(f'models/{lang1}_model.pkl')
    mlp_de = joblib.load(f'models/{lang2}_model.pkl')

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

    # Predict RGB values using the MLP model
    def predict_rgb(mlp_model, concept_vectors):
        return mlp_model.predict(concept_vectors)

    rgb_values_en = predict_rgb(mlp_en, concept_vectors1)
    rgb_values_de = predict_rgb(mlp_de, concept_vectors2)
    # add the minimum value to all values to avoid negative values for each row
    # rgb_values_en = rgb_values_en + np.abs(np.min(rgb_values_en, axis=1))
    # rgb_values_de = rgb_values_de + np.abs(np.min(rgb_values_de, axis=1))

    def rgb_to_hex(rgb):
        # Ensure the RGB values are within the range [0, 255]
        min_val = abs(min(rgb))
        rgb = rgb + min_val
        r, g, b = [int(max(0, min(255, x * 255))) for x in rgb]
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    # Print the RGB values for each concept
    print("Concept\tEnglish RGB")
    for concept, rgb_en in zip(concepts_filtered1, rgb_values_en):
        hex_en = rgb_to_hex(rgb_en)
        print(f"{concept}\t{hex_en}")
    print("Concept\tOther RGB")
    for concept, rgb_de in zip(concepts_filtered2, rgb_values_de):
        hex_en = rgb_to_hex(rgb_de)
        print(f"{concept}\t{hex_en}")


if __name__ == '__main__':
    main()