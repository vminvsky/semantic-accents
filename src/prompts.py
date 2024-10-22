import pandas as pd 
import os 
from dataclasses import dataclass, field
from datasets import Dataset

from utils import open_file

@dataclass 
class CulturalBenchDataset(Dataset):
    lang: str 
    difficulty: str = field(default='easy') # 'easy' or 'hard'
    model_name: str = field(default='gpt-4o')
    include_country: bool = field(default=False)
    randomize_options: bool = field(default=False)
    
    # dataframe 
    df: pd.DataFrame = field(default=None)
    col_mapping: dict = field(default_factory=dict)
    data_dir: str = field(default=os.path.join('data', 'cultural_bench'))
    data_path: str = field(default=None)
    options: list = field(default_factory=lambda: ['A', 'B', 'C', 'D'])
    drop_nans: bool = field(default=True) # NOTE: THIS IS BAD TRANSLATIONS. FIXXXX!

    # loading files 
    prompt_template: str = field(default=None)
    prompt_template_path: str = field(default=os.path.join(
                "data", "prompts", "{lang}", "cultural_bench_{difficulty}.txt"))

    def __post_init__(self):
        if self.randomize_options is True:
            raise NotImplementedError("Randomizing options is not yet supported")
        
        if self.difficulty == 'hard':
            raise NotImplementedError("Hard difficulty is not yet supported")

        self._create_col_mapping()
        if self.lang == 'en':
            self.data_path = os.path.join(self.data_dir, self.model_name, 'no_country.jsonl')
            
        else: 
            self.data_path = os.path.join(self.data_dir, self.model_name, self.lang, f'data.jsonl')

        self.df = pd.read_json(self.data_path, lines=True)
        if self.drop_nans:
            self.df.dropna(inplace=True)

        if self.prompt_template is None:
            self.prompt_template = open_file(self.prompt_template_path.format(lang=self.lang, difficulty=self.difficulty))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.difficulty == 'easy':
            prompt = self._easy_formatting(row)
            return prompt

    def _easy_formatting(self, row):
        # formats a row depending on the necessary stuff
        prompt_template = self.prompt_template

        if self.include_country is True:
            question_col = self.col_mapping['question']
        else:
            question_col = self.col_mapping['question_no_country']
        
        question = row[question_col]
        question_idx = row[self.col_mapping['question_idx']]
        question = prompt_template.format(question=question)

        answer = row[self.col_mapping['answer']]
        country = row[self.col_mapping['country']]

        options = self._return_options(row)
        prompt = self._add_options(question, options)
        return prompt, answer, country, question_idx

    def _return_options(self, row):
        # converts option cols to list, simply silly helper
        options = [row[self.col_mapping[f'option_{option}']] for option in ['a', 'b', 'c', 'd']]
        return options
    
    def _add_options(self, prompt, options):
        # adds options to the prompt in the correct format
        return prompt + '\n' + '\n'.join([f"{char}: {option}" for char, option in zip(self.options, options)])

    def _create_col_mapping(self):
        self.col_mapping = {'country': 'country', 'answer': 'answer', 'question_idx': 'question_idx'}
        if self.lang == 'en':
            self.col_mapping.update({
                'question_no_country': 'prompt_no_country',
                'question': 'prompt_question',
            })
            for option in ['a', 'b', 'c', 'd']:
                self.col_mapping[f'option_{option}'] = f'prompt_option_{option}'
        else: 
            self.col_mapping.update({
                'question_no_country': f'prompt_no_country_{self.lang}',
                'question': f'prompt_question_{self.lang}',
            })
            for option in ['a', 'b', 'c', 'd']:
                self.col_mapping[f'option_{option}'] = f'prompt_option_{option}_{self.lang}'

    def __len__(self):
        return len(self.df)
    
def main():
    dataset = CulturalBenchDataset(lang='en', include_country=True)
    print(dataset[10])

if __name__ == "__main__":
    main()