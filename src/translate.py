import os
import openai
import asyncio
import backoff
import pandas as pd
from datasets import load_dataset

from utils import country_languages

# Set up the OpenAI Azure async client
client = openai.AsyncAzureOpenAI(
    api_key="e36dea00403e498382f4f606c027e4c9",
    azure_endpoint="https://api-ai-sandbox.princeton.edu",
    api_version="2024-02-01"
)

# Backoff on rate limit errors
@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def make_api_call_to_gpt(prompt, model="gpt-4o"):
    messages = [{"role": "user", "content": prompt}]
    try: 
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""

# Function to limit the number of concurrent API calls
async def limited_api_call(semaphore, prompt, model="gpt-4o"):
    async with semaphore:
        return await make_api_call_to_gpt(prompt, model=model)

# Define the prompt formatting logic
remove_country_prefix = "Rewrite the following text from English to {language}:\n\n{question}"

def process_question(question: str, language: str):
    return remove_country_prefix.format(question=question, language=language)

# Main async function to handle the dataset and API calls
async def main():
    debug = True 
    MODEL_NAME = 'GPT-4o' # "Meta-Llama-3-1-70B-Instruct-htzs", 'GPT-4o'
    data_dir = f'data/cultural_bench/{MODEL_NAME}'
    cols_to_translate_over = ['prompt_question', 'prompt_no_country', 
                              'prompt_option_a', 'prompt_option_b', 
                              'prompt_option_c', 'prompt_option_d']
    
    # for i, country in enumerate(country_languages.keys()):
    for i, country in enumerate(['Germany', 'Japan', 'Russia']):
        data = pd.read_json(os.path.join(data_dir, 'no_country.jsonl'), lines=True)
        if debug: 
            data = data.head(50)
        for col in cols_to_translate_over:
            # Cap the number of concurrent requests
            lang_name = country_languages[country][0]
            lang_id = country_languages[country][1]

            max_concurrent_requests = 40
            semaphore = asyncio.Semaphore(max_concurrent_requests)

            prompts = [process_question(q, lang_name) for q in data[col]]

            # Create a list to store the results of asynchronous calls
            tasks = [limited_api_call(semaphore, prompt, model=MODEL_NAME) for prompt in prompts]

            # Gather and run the tasks concurrently
            results = await asyncio.gather(*tasks)

            # Add the results to a new column in the DataFrame
            data[f'{col}_{lang_id}'] = results

        # Save the updated dataset locally
        save_path = os.path.join(data_dir, f'{lang_id}/data.jsonl')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_json(save_path, force_ascii=False, lines=True, orient='records')
        print(f"{lang_name} finished!")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
