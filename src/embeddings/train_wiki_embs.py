import os
import argparse
import json
from typing import Union
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from gensim.models import FastText
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess
import spacy
# import fugashi
from datetime import datetime
import pickle
from dataclasses import dataclass
from tqdm import tqdm

import sys
sys.path.append('src/')
from config import langs


@dataclass
class TrainingConfig:
    vector_size: int = 100
    window: int = 5
    negative: int = 10
    min_count: int = 5
    lr: float = 0.025
    workers: int = 10
    sg: int = 1  # Skip-gram is better for parallelogram tasks
    epochs: int = 5
    seed: int = 42
    min_n: int = 0
    max_n: int = 0
    shrink_windows: bool = True

# Create an instance of TrainingConfig
config = TrainingConfig()

class WikiEmbeddingTrainer:
    def __init__(self, config, base_dir: str = 'data/wiki_gen'):
        """Initialize the embedding trainer."""
        self.base_dir = Path(base_dir)
        self.config = config
        self.setup_logging()
        self.setup_tokenizers()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'embedding_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_tokenizers(self):
        """Initialize tokenizers for different languages."""
        self.tokenizers = {}
        self.tokenizer_versions = {}
        try:
            # English
            self.tokenizers['en'] = spacy.load('en_core_web_sm')
            self.tokenizer_versions['en'] = self.tokenizers['en'].meta.get('version', 'unknown')
        except:
            self.logger.warning("English SpaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.tokenizer_versions['en'] = None
            
        try:
            # German
            self.tokenizers['de'] = spacy.load('de_core_news_sm')
            self.tokenizer_versions['de'] = self.tokenizers['de'].meta.get('version', 'unknown')
        except:
            self.logger.warning("German SpaCy model not found. Run: python -m spacy download de_core_news_sm")
            self.tokenizer_versions['de'] = None
            
        try:
            # Russian
            self.tokenizers['ru'] = spacy.load('ru_core_news_sm')
            self.tokenizer_versions['ru'] = self.tokenizers['ru'].meta.get('version', 'unknown')
        except:
            self.logger.warning("Russian SpaCy model not found. Run: python -m spacy download ru_core_news_sm")
            self.tokenizer_versions['ru'] = None
            
        try:
            # Estonian - using simple tokenizer as spaCy doesn't have Estonian
            self.tokenizers['et'] = lambda text: simple_preprocess(text, deacc=True)
            self.tokenizer_versions['et'] = 'simple_preprocess'
        except:
            self.logger.warning("Using simple tokenizer for Estonian")
            self.tokenizer_versions['et'] = None
            
        try:
            # Japanese
            import fugashi
            self.tokenizers['ja'] = fugashi.Tagger()
            self.tokenizer_versions['ja'] = fugashi.__version__
        except:
            self.logger.warning("Japanese Fugashi tagger not found. Install fugashi and mecab-python3")
            self.tokenizer_versions['ja'] = None

    def tokenize_text(self, text: str, language: str, remove_stopwords: bool = False) -> List[str]:
        """Tokenize text based on language."""
        if language not in self.tokenizers:
            self.logger.warning(f"No specific tokenizer for {language}, using simple tokenizer")
            return simple_preprocess(text, deacc=True)
            
        if language in ['ja']:
            # Special handling for Japanese
            return [token.surface for token in self.tokenizers[language].parse(text).split()]
        elif language in ['en', 'de', 'ru', 'fr', 'hi']:
            # SpaCy-based tokenization
            doc = self.tokenizers[language](text)
            if remove_stopwords:
                return [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
            else:
                return [token.text.lower() for token in doc if not token.is_punct]
        elif language == 'et':
            return self.tokenizers[language](text)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def preprocess_articles(self, articles: List[Dict], language: str) -> List[List[str]]:
        """Preprocess articles into tokenized sentences."""
        processed_articles = []
        
        for article in tqdm(articles, desc="formatting articles"):
            tokens = self.tokenize_text(article['article'], language)
            processed_articles.append(tokens)
            
        return processed_articles

    def train_embeddings(self, articles: List[Dict], language: str, 
                         model_name: str, lang_dir: Path, article_files_info: Dict[str, float], with_bigrams: bool = False) -> FastText:
        """Train FastText embeddings on the articles."""
        self.logger.info(f"Training embeddings for {language} - {model_name}")
        
        # Path to the cached processed articles
        processed_articles_file = lang_dir / "processed_articles.pkl"

        if processed_articles_file.exists():
            self.logger.info(f"Loading processed articles from {processed_articles_file}")
            with open(processed_articles_file, "rb") as f:
                data = pickle.load(f)
            processed_articles = data['processed_articles']
            metadata = data['metadata']

            # Check if articles or tokenizer have changed
            if metadata['article_files_info'] == article_files_info and \
               metadata['tokenizer_version'] == self.tokenizer_versions.get(language, None):
                self.logger.info("No changes in articles or tokenizer detected. Using cached processed articles.")
            else:
                self.logger.info("Changes in articles or tokenizer detected. Reprocessing articles.")
                processed_articles = self.preprocess_articles(articles, language)
                # Save processed_articles and metadata
                data = {
                    'processed_articles': processed_articles,
                    'metadata': {
                        'article_files_info': article_files_info,
                        'tokenizer_version': self.tokenizer_versions.get(language, None)
                    }
                }
                with open(processed_articles_file, "wb") as f:
                    pickle.dump(data, f)
        else:
            self.logger.info("No cached processed articles found. Processing articles.")
            processed_articles = self.preprocess_articles(articles, language)
            # Save processed_articles and metadata
            data = {
                'processed_articles': processed_articles,
                'metadata': {
                    'article_files_info': article_files_info,
                    'tokenizer_version': self.tokenizer_versions.get(language, None)
                }
            }
            with open(processed_articles_file, "wb") as f:
                pickle.dump(data, f)
        
        if with_bigrams:
            self.logger.info(f"Building bigrams...") # NOTE this is probably useless. 
            phrases = Phrases(processed_articles, min_count=20, threshold=20)
            bigram = Phraser(phrases)
            # Apply bigram transformation
            processed_articles = [bigram[article] for article in processed_articles]
        else:
            bigram = None

        # Train FastText model
        model = FastText(
            vector_size=self.config.vector_size,
            window=self.config.window,
            negative=self.config.negative,
            min_count=self.config.min_count,
            workers=self.config.workers,
            sg=self.config.sg,  # Skip-gram
            epochs=self.config.epochs,
            seed=self.config.seed,
            min_n=self.config.min_n,
            max_n=self.config.max_n,
            alpha=self.config.lr,
            shrink_windows=self.config.shrink_windows,
)
        self.logger.info(f"Building vocab...")
        model.build_vocab(corpus_iterable=processed_articles)

        self.logger.info(f"Training...")
        model.train(
            corpus_iterable=processed_articles,
            total_examples=len(processed_articles),
            epochs=model.epochs,
            compute_loss=True,
        )
        
        return model, bigram

    def save_models(self, model: FastText, language: str, 
                   model_name: str, output_dir: Path, bigram: Phraser=None):
        """Save trained models and related artifacts."""
        # Create output directory
        model_dir = output_dir / f"{model_name}_{language}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FastText model
        model.save(str(model_dir / "fasttext.model"))
        
        if bigram:
            # Save bigram phraser
            with open(model_dir / "bigram.pkl", "wb") as f:
                pickle.dump(bigram, f)
        
        # Save model metadata
        metadata = {
            "language": language,
            "model_name": model_name,
            "vector_size": model.vector_size,
            "window": model.window,
            "epochs": model.epochs,
            "training_date": datetime.now().isoformat()
        }
        
        with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def process_all_articles(self, output_dir: Path = Path("embeddings"), 
                             limit_langs: list = ['en','de','fr','ru','ja', 'et'],
                             with_bigrams: bool = False):
        """Process all articles and train embeddings for each language and model."""
        # Iterate through all model directories
        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                
                # Iterate through language directories
                for lang_dir in model_dir.iterdir():
                    if lang_dir.is_dir():
                        language = lang_dir.name
                        if language not in limit_langs:
                            continue
                        self.logger.info(f"Processing {model_name} - {language}")
                        
                        # Read all JSONL files in the directory
                        articles = []
                        article_files_info = {}
                        for jsonl_file in lang_dir.glob("*.jsonl"):
                            modified_time = jsonl_file.stat().st_mtime
                            article_files_info[str(jsonl_file)] = modified_time
                            df = pd.read_json(jsonl_file, lines=True)
                            articles.extend(df.to_dict('records'))

                        if articles:
                            # Train embeddings
                            model, bigram = self.train_embeddings(
                                articles, language, model_name, lang_dir, article_files_info, with_bigrams=with_bigrams)
                            
                            # Save models and artifacts
                            self.save_models(model=model, bigram=bigram, language=language, model_name=model_name, output_dir=output_dir)
                        else:
                            self.logger.warning(f"No articles found for {model_name} - {language}")

def parse_languages(value):
    return value.split(',') if ',' in value else [value]

def main():
    parser = argparse.ArgumentParser(description="Train Wiki Embeddings")
    parser.add_argument('--lang', type=parse_languages, required=True, help='Language(s) for training (comma-separated for multiple)')
    args = parser.parse_args()
    config = TrainingConfig()
    trainer = WikiEmbeddingTrainer(config)
    trainer.process_all_articles(limit_langs=args.lang)

if __name__ == "__main__":
    main()