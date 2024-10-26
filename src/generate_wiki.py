import os
import json
import asyncio
import aiofiles
from openai import AzureOpenAI
import tiktoken
from typing import List, Dict
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ArticleRequest:
    title: str
    content: str
    lang: str
    max_words: int = 2000

@dataclass
class ArticleResponse:
    title: str
    content: str
    lang: str
    article: str
    prompt: str
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    model: str
    timestamp: str
    status: str
    error: str = None

class AsyncWikiArticleGenerator:
    def __init__(self, api_key: str, endpoint: str, deployment: str, max_concurrent: int = 5):
        """Initialize the generator with Azure OpenAI credentials."""
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=endpoint
        )
        self.deployment = deployment
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_concurrent = max_concurrent
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_output_directory(self, model: str, language: str) -> Path:
        """Create and return the output directory path."""
        output_dir = Path('data/wiki_gen') / model / language
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    async def generate_single_article(self, request: ArticleRequest) -> ArticleResponse:
        """Generate a single article with the given parameters."""
        try:
            target_tokens = request.max_words
            
            # Create the prompt based on the language
            prompt_template = {
                "en": """Write a complete Wikipedia-style article in English about "{title}" following this outline:

{content}

Requirements:
1. Write the article in a neutral, encyclopedic tone
2. Include an introduction before the first section
3. Follow the exact outline structure using "#" for main sections and "##" for subsections
4. Do not include any citations or references
5. Make the article approximately {max_words} words in length
6. Ensure smooth transitions between sections
7. Include relevant details and explanations in each section
8. Maintain consistent style throughout the article
9. Keep the content factual and informative

Write the complete article now:""",
                "de": """Schreibe einen vollständigen Wikipedia-Artikel auf Deutsch über "{title}" nach dieser Gliederung:

{content}

Anforderungen:
1. Schreibe den Artikel in einem neutralen, enzyklopädischen Ton
2. Füge eine Einleitung vor dem ersten Abschnitt ein
3. Folge der exakten Gliederungsstruktur mit "#" für Hauptabschnitte und "##" für Unterabschnitte
4. Füge keine Zitate oder Referenzen ein
5. Der Artikel sollte ungefähr {max_words} Wörter lang sein
6. Stelle flüssige Übergänge zwischen den Abschnitten sicher
7. Füge relevante Details und Erklärungen in jeden Abschnitt ein
8. Behalte einen einheitlichen Stil im gesamten Artikel bei
9. Halte den Inhalt sachlich und informativ

Schreibe jetzt den vollständigen Artikel:""",
"ru": """Напишите полноценную статью в стиле Википедии на русском языке о "{title}", следуя этому плану:

{content}

Требования:
1. Напишите статью в нейтральном, энциклопедическом тоне
2. Включите введение перед первым разделом
3. Следуйте точной структуре плана, используя "#" для основных разделов и "##" для подразделов
4. Не включайте цитаты или ссылки
5. Статья должна быть примерно {max_words} слов в длину
6. Обеспечьте плавные переходы между разделами
7. Включите важные детали и пояснения в каждый раздел
8. Поддерживайте единый стиль на протяжении всей статьи
9. Сохраняйте фактическую точность и информативность

Напишите полную статью сейчас:""",

                "et": """Kirjutage täielik Vikipeedia-stiilis artikkel eesti keeles teemal "{title}", järgides seda struktuuri:

{content}

Nõuded:
1. Kirjutage artikkel neutraalses, entsüklopeedilises toonis
2. Lisage sissejuhatus enne esimest jaotist
3. Järgige täpset struktuuri, kasutades "#" põhijaotiste ja "##" alamjaotiste jaoks
4. Ärge lisage tsitaate ega viiteid
5. Artikkel peaks olema umbes {max_words} sõna pikk
6. Tagage sujuvad üleminekud jaotiste vahel
7. Lisage igasse jaotisse olulised üksikasjad ja selgitused
8. Säilitage ühtne stiil kogu artikli vältel
9. Hoidke sisu faktipõhine ja informatiivne

Kirjutage nüüd täielik artikkel:""",

                "ja": """『{title}』についての Wikipedia スタイルの記事を日本語で、以下の構成に従って書いてください：

{content}

要件：
1. 中立的で百科事典的な文体で書く
2. 最初のセクションの前に導入部を含める
3. "#" を主要セクション、"##" をサブセクションとして、正確な構成に従う
4. 引用や参考文献を含めない
5. 記事は約 {max_words} 語程度の長さにする
6. セクション間の移行をスムーズにする
7. 各セクションに関連する詳細と説明を含める
8. 記事全体で一貫したスタイルを維持する
9. 内容は事実に基づき、情報価値の高いものにする

それでは、完全な記事を書いてください："""
            }

            prompt = prompt_template.get(request.lang).format(
                title=request.title,
                content=request.content,
                max_words=request.max_words
            )

            # Generate the article
            response = await asyncio.get_event_loop().run_in_executor(
                ThreadPoolExecutor(),
                lambda: self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=target_tokens,
                )
            )
            
            article = response.choices[0].message.content.strip()
        

            return ArticleResponse(
                title=request.title,
                content=request.content,
                lang=request.lang,
                article=article,
                prompt=prompt,
                completion_tokens=response.usage.completion_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
                model=self.deployment,
                timestamp=datetime.now().isoformat(),
                status="success"
            )

        except Exception as e:
            self.logger.error(f"Error generating article {request.title}: {str(e)}")
            return ArticleResponse(
                title=request.title,
                content=request.content,
                lang=request.lang,
                article="",
                prompt=prompt,
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
                model=self.deployment,
                timestamp=datetime.now().isoformat(),
                status="error",
                error=str(e)
            )

    async def process_batch(self, requests: List[ArticleResponse]) -> List[ArticleResponse]:
        """Process a batch of article requests concurrently."""
        tasks = [self.generate_single_article(req) for req in requests]
        return await asyncio.gather(*tasks)

    async def save_results(self, results: List[ArticleResponse], output_dir: Path):
        """Save results to a JSONL file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f'articles_{timestamp}.jsonl'
        
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                await f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')

    async def process_jsonl_file(self, input_file: str):
        """Process all articles from a JSONL file."""
        
        # Read the JSONL file
        df = pd.read_json(input_file, lines=True)
        
        # Group requests by language
        language_groups = df.groupby('lang')
        
        for language, group in language_groups:
            
            # Create output directory
            output_dir = self.create_output_directory(self.deployment, language)
            
            # Convert DataFrame rows to ArticleRequest objects
            requests = [
                ArticleRequest(
                    title=row.title,
                    content=row.content,
                    lang=language,
                    max_words=4096
                )
                for _, row in group.iterrows()
            ]

            # sample 500 random requests
            import random
            requests = random.sample(requests, min(3, len(requests)))

            self.logger.info(f"Processing {len(requests)} articles for {language}...")

            
            # Process requests in batches
            for i in range(0, len(requests), self.max_concurrent):
                batch = requests[i:i + self.max_concurrent]
                self.logger.info(f"Processing batch {i//self.max_concurrent + 1} of {len(requests)//self.max_concurrent + 1}")
                
                results = await self.process_batch(batch)
                await self.save_results(results, output_dir)
                
                # Add a small delay between batches
                await asyncio.sleep(1)

async def main():
    generator = AsyncWikiArticleGenerator(
        api_key="",
        endpoint="https://api-ai-sandbox.princeton.edu",
        deployment="Meta-Llama-3-1-70B-Instruct-htzs",
        max_concurrent=50
    )
    
    await generator.process_jsonl_file('data/wiki_download/en/data.jsonl')

if __name__ == "__main__":
    asyncio.run(main())