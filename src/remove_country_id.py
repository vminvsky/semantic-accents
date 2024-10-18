import os
import openai
import asyncio
import backoff
import pandas as pd
from datasets import load_dataset

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
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

# Function to limit the number of concurrent API calls
async def limited_api_call(semaphore, prompt, model="gpt-4o"):
    async with semaphore:
        return await make_api_call_to_gpt(prompt, model=model)

# Define the prompt formatting logic
remove_country_prefix = "From the following question, remove any mention of a country. Rewrite it as if it were a general question. Here is an example:\n\n \
'In the Netherlands, which of the following is an unusual common public practice?' -> 'Which of the following is an unusual common public practice?'\n\
'In Philippine culture, which of the following is considered rude to ask someone, especially if you are not related?' -> 'Which of the following is considered rude to ask someone, especially if you are not related?'\n\n\
Make sure not to include any processing and respond just with the formatted question.\
\n\n{question}"

def process_question(question):
    return remove_country_prefix.format(question=question)

# Main async function to handle the dataset and API calls
async def main():
    MODEL_NAME = "gpt-4o"
    # Load dataset
    ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Easy")
    data = pd.DataFrame(ds['test'])
    
    # Cap the number of concurrent requests
    max_concurrent_requests = 20
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    prompts = [process_question(q) for q in data['prompt_question']]

    # Create a list to store the results of asynchronous calls
    tasks = [limited_api_call(semaphore, prompt, model=MODEL_NAME) for prompt in prompts]

    # Gather and run the tasks concurrently
    results = await asyncio.gather(*tasks)

    # Add the results to a new column in the DataFrame
    data['prompt_no_country'] = results

    # Save the updated dataset locally
    save_path = f'data/cultural_bench/{MODEL_NAME}/no_country.jsonl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.to_json(save_path, force_ascii=False, lines=True, orient='records')
    print("Dataset with OpenAI responses saved successfully!")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
