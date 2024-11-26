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
from transformers import AutoTokenizer
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
    negative: int = 5
    min_count: int = 5
    lr: float = 0.025
    workers: int = 1
    sg: int = 1  # Skip-gram is better for parallelogram tasks
    epochs: int = 10
    seed: int = 42
    min_n: int = 5
    max_n: int = 5
    shrink_windows: bool = True

@dataclass 
class DataMapping:
    synth_wiki: Path = Path('data/wiki_gen_cleaned/')
    real_wiki: Path = Path('data/wikipedia/sections/cleaned/')
    synth_bedtime: Path = Path('data/bedtime_stories_cleaned/')


# Create an instance of TrainingConfig
config = TrainingConfig()
data_mapping = DataMapping()

class WikiEmbeddingTrainer:
    def __init__(self, config, data_source: str = 'synth-wiki', tokenizer: str = 'hf', overwrite=True):
        """Initialize the embedding trainer."""
        self.data_source = data_source 
        self.overwrite = overwrite
        self.tokenizer = tokenizer
        self.langs = ['bn', 'de', 'en', 'et', 'fr', 'hi', 'ja', 'ru']
        self.base_dir = getattr(data_mapping, self.data_source.replace('-','_'))
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
        if self.tokenizer == 'hf':
            for lang in self.langs:
                try:
                    self.tokenizers[lang] = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
                    self.tokenizer_versions[lang] = 'meta-llama/Meta-Llama-3-8B'
                except:
                    self.logger.warning(f"Tokenizer not found for {lang}")
                    self.tokenizer_versions[lang] = None
        elif self.tokenizer == 'spacy':
            raise NotImplementedError("Spacy tokenizers are not supported yet.")

    def tokenize_text(self, text: str, language: str, remove_stopwords: bool = False) -> List[str]:
        """Tokenize text based on language."""
        # TODO debate creating language-specific tokenizers. will be important for symbol langs like jap
        return text.split()

    def preprocess_articles(self, articles: List[Dict], language: str) -> List[List[str]]:
        """Preprocess articles into tokenized sentences."""
        processed_articles = []
        
        total_words = 0
        for article in tqdm(articles, desc="formatting articles"):
            tokens = self.tokenize_text(article['article'], language)
            total_words += len(tokens)
            processed_articles.append(tokens)
        print('total words', total_words)
        return processed_articles

    def train_embeddings(self, articles: List[Dict], language: str, 
                         model_name: str, lang_dir: Path, article_files_info: Dict[str, float], with_bigrams: bool = False) -> FastText:
        """Train FastText embeddings on the articles."""
        self.logger.info(f"Training embeddings for {language} - {model_name}")
        
        # Path to the cached processed articles
        processed_articles_file = lang_dir / "processed_articles.pkl"

        if (processed_articles_file.exists()) and not self.overwrite:
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
                             limit_langs: list = ['en','de','fr','ru','ja', 'et', 'bn', 'hi'],
                             with_bigrams: bool = False):
        """Process all articles and train embeddings for each language and model."""
        # Iterate through all model directories
        output_dir = output_dir / self.data_source
        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name

                if '70B' in model_name:
                    continue
                
                # Iterate through language directories
                for lang_dir in model_dir.iterdir():
                    print(lang_dir)
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

                        # self.logger.warning('LENGTH ARTICLES', len(articles))

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
    parser.add_argument('--data_source', type=str, required=True, help='Which data source to use (`synth-wiki` or `real-wiki` or `synth-bedtime`)')
    parser.add_argument('--overwrite', default=False, help='Overwrite the existing processed articles.')
    args = parser.parse_args()
    config = TrainingConfig()
    trainer = WikiEmbeddingTrainer(config,data_source=args.data_source, overwrite=args.overwrite)
    trainer.process_all_articles(limit_langs=args.lang)

if __name__ == "__main__":
    main()