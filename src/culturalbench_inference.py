import os
import openai
import asyncio
import backoff
import pandas as pd
from datasets import load_dataset
import json
from typing import List
from prompts import CulturalBenchDataset
import argparse
from tqdm.asyncio import tqdm_asyncio
import weave


# Set up the OpenAI Azure async client
client = openai.AsyncAzureOpenAI(
    api_key="",
    azure_endpoint="https://api-ai-sandbox.princeton.edu",
    api_version="2024-05-01-preview"
)

# Backoff on rate limit errors
@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def make_api_call_to_gpt(prompt, model="gpt-4o") -> List:
    messages = [{"role": "user", "content": prompt}]
    try: 
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5
        )
        return {t.token: t.logprob for t in response.choices[0].logprobs.content[0].top_logprobs}
    except Exception as e:
        print(e)
        return ""

# Function to limit the number of concurrent API calls
async def limited_api_call(semaphore, prompt, answer, country, q_id, model="gpt-4o") -> List:
    async with semaphore:
        return await make_api_call_to_gpt(prompt, model=model), answer, country, q_id

# Define the prompt formatting logic
remove_country_prefix = "Rewrite the following text from English to {language}:\n\n{question}"

def process_question(question: str, language: str):
    return remove_country_prefix.format(question=question, language=language)

async def main(model, language, include_country):

    LANGUAGE = language
    INCLUDE_COUNTRY = include_country

    # Load the dataset
    dataset = CulturalBenchDataset(lang=LANGUAGE,
                                   include_country=INCLUDE_COUNTRY)
        
    MODEL_NAME = 'gpt-4o' # "Meta-Llama-3-1-70B-Instruct-htzs", 'GPT-4o'
    data_dir = f'data/cultural_bench/{MODEL_NAME}'

    data = []

    max_concurrent_requests = 40
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    # Create a list to store the results of asynchronous calls
    tasks = [limited_api_call(semaphore, dataset[i][0], dataset[i][1], dataset[i][2], dataset[i][3], model=MODEL_NAME) for i in range(len(dataset))]

    # Gather and run the tasks concurrently
    results = await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Processing")

    # add results to the jsonl file
    for i, result in enumerate(results):
        item = {
            "logprobs": result[0],
            "answer": result[1],
            "country": result[2],
            "question_idx": int(result[3]),
        }
        data.append(item)


    # Write the data to jsonl file
    os.makedirs(data_dir, exist_ok=True)
    country_include = 'with_country' if INCLUDE_COUNTRY else 'no_country'
    with open(f'{data_dir}/{LANGUAGE}/eval_results_{country_include}_{model}.jsonl', 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# Run the main function
if __name__ == "__main__":

    # specify command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='de', help='language to translate to')
    parser.add_argument('--include_country', type=bool, default=False, help='whether to include country in the prompt')
    parser.add_argument('--model', type=str, default='gpt-4o', help='model to use for inference')
    args = parser.parse_args()

    # weave.init(f"culturalbench_inference_{args.lang}_{args.include_country}")

    asyncio.run(main(args.model, args.lang, args.include_country))
