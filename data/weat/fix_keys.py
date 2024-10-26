"""
Most of the WEAT code in this repository was taken from https://github.com/chadaeun/weat_replication/tree/master
"""
import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def replace_keys(target_dict, reference_dict):
    new_dict = {}
    for ref_key, target_key in zip(reference_dict.keys(), target_dict.keys()):
        if isinstance(target_dict[target_key], dict) and isinstance(reference_dict[ref_key], dict):
            new_dict[ref_key] = replace_keys(target_dict[target_key], reference_dict[ref_key])
        else:
            new_dict[ref_key] = target_dict[target_key]
    return new_dict

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    weat = load_json('weat_weird_keys.json')
    en = weat['English']
    
    for lang in weat:
        if lang != 'English':
            modified_dict = replace_keys(weat[lang], en)
            save_json(modified_dict, f'{lang.lower()}.json')
    
    save_json(en, 'english.json')

if __name__ == '__main__':
    main()