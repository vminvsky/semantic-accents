import os
import json
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


@dataclass
class TrainingConfig:
    vector_size: int = 300
    window: int = 5
    negative: int = 10
    min_count: int = 3
    workers: int = 4
    sg: int = 1  # Skip-gram is better for parallelogram tasks
    epochs: int = 20
    seed: int = 42
    # min_n: int = 5
    # max_n: int = 5

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
        try:
            # English
            self.tokenizers['en'] = spacy.load('en_core_web_sm')
        except:
            self.logger.warning("English SpaCy model not found. Run: python -m spacy download en_core_web_sm")
            
        try:
            # German
            self.tokenizers['de'] = spacy.load('de_core_news_sm')
        except:
            self.logger.warning("German SpaCy model not found. Run: python -m spacy download de_core_news_sm")
            
        try:
            # Russian
            self.tokenizers['ru'] = spacy.load('ru_core_news_sm')
        except:
            self.logger.warning("Russian SpaCy model not found. Run: python -m spacy download ru_core_news_sm")
            
        try:
            # Estonian - using simple tokenizer as spaCy doesn't have Estonian
            self.tokenizers['et'] = lambda text: simple_preprocess(text, deacc=True)
        except:
            self.logger.warning("Using simple tokenizer for Estonian")
            
        try:
            # Japanese
            self.tokenizers['ja'] = fugashi.Tagger()
        except:
            self.logger.warning("Japanese Fugashi tagger not found. Install fugashi and mecab-python3")

    def tokenize_text(self, text: str, language: str) -> List[str]:
        """Tokenize text based on language."""
        if language not in self.tokenizers:
            self.logger.warning(f"No specific tokenizer for {language}, using simple tokenizer")
            return simple_preprocess(text, deacc=True)
            
        if language == 'ja':
            # Special handling for Japanese
            return [token.surface for token in self.tokenizers[language].parse(text).split()]
        elif language in ['en', 'de', 'ru']:
            # SpaCy-based tokenization
            doc = self.tokenizers[language](text)
            return [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
        else:
            raise ValueError(f"Unsupported language: {language}")

    def preprocess_articles(self, articles: List[Dict], language: str) -> List[List[str]]:
        """Preprocess articles into tokenized sentences."""
        processed_articles = []
        
        for article in articles:
            # Process main text
            tokens = self.tokenize_text(article['article'], language)
            processed_articles.append(tokens)
            
        return processed_articles

    def train_embeddings(self, articles: List[Dict], language: str, model_name: str) -> FastText:
        """Train FastText embeddings on the articles."""
        self.logger.info(f"Training embeddings for {language} - {model_name}")
        
        # Preprocess articles
        processed_articles = self.preprocess_articles(articles, language)
        
        # Train phrases (bigrams)
        phrases = Phrases(processed_articles)
        bigram = Phraser(phrases)
        
        # Apply bigram transformation
        processed_articles = [bigram[article] for article in processed_articles]
        
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
)
        
        # Build vocabulary
        model.build_vocab(corpus_iterable=processed_articles)
        
        # Train the model
        model.train(
            corpus_iterable=processed_articles,
            total_examples=len(processed_articles),
            epochs=model.epochs
        )
        
        return model, bigram

    def save_models(self, model: FastText, bigram: Phraser, language: str, 
                   model_name: str, output_dir: Path):
        """Save trained models and related artifacts."""
        # Create output directory
        model_dir = output_dir / f"{model_name}_{language}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FastText model
        model.save(str(model_dir / "fasttext.model"))
        
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

    def process_all_articles(self, output_dir: Path = Path("embeddings")):
        """Process all articles and train embeddings for each language and model."""
        # Iterate through all model directories
        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                
                # Iterate through language directories
                for lang_dir in model_dir.iterdir():
                    if lang_dir.is_dir():
                        language = lang_dir.name
                        self.logger.info(f"Processing {model_name} - {language}")
                        
                        # Read all JSONL files in the directory
                        articles = []
                        for jsonl_file in lang_dir.glob("*.jsonl"):
                            df = pd.read_json(jsonl_file, lines=True)
                            articles.extend(df.to_dict('records'))
                        
                        if articles:
                            # Train embeddings
                            model, bigram = self.train_embeddings(articles, language, model_name)
                            
                            # Save models and artifacts
                            self.save_models(model, bigram, language, model_name, output_dir)
                        else:
                            self.logger.warning(f"No articles found for {model_name} - {language}")

def main():
    config = TrainingConfig()
    trainer = WikiEmbeddingTrainer(config)
    trainer.process_all_articles()

if __name__ == "__main__":
    main()