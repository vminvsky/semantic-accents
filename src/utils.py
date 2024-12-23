
def open_file(file_path):
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            return f.readlines()
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            return f.read()

def printv(*args, verbose=0):
    yellow = "\033[93m"
    reset = "\033[0m"
    
    if verbose == 1:
        for value in args:
            print(f"{yellow}{value}{reset}")


def load_top_words(path='data/1000_words_ef.txt'):
    # load dataset from EF top 1000 used words 
    with open(path, 'r') as f:
        words = f.readlines()
    return [word.strip() for word in words]

ctry_lang = {
    'nl': 'Netherlands',
    'ko': 'South Korea',
    'en': 'United Kingdom',
    'es': 'Spain',
    'ar': 'Saudi Arabia',
    'hi': 'India',
    'vi': 'Vietnam',
    'ms': 'Malaysia',
    'ja': 'Japan',
    'fa': 'Iran',
    'fil': 'Philippines',
    'he': 'Israel',
    'bn': 'Bangladesh',
    'uk': 'Ukraine',
    'cs': 'Czech Republic',
    'it': 'Italy',
    'id': 'Indonesia',
    'fr': 'France',
    'ne': 'Nepal',
    'ro': 'Romania',
    'de': 'Germany',
    'ur': 'Pakistan',
    'pt': 'Brazil',
    'ru': 'Russia',
    'zh': 'China',
    'zh-yue': 'Hong Kong',
    'tr': 'Turkey',
    'pl': 'Poland',
    'th': 'Thailand'
}

country_languages = {
    'Netherlands': ['Dutch', 'nl'],
    'South Korea': ['Korean', 'ko'],
    'Nigeria': ['English', 'en'],
    'Spain': ['Spanish', 'es'],
    'Morocco': ['Arabic', 'ar'],
    'India': ['Hindi', 'hi'],
    'United Kingdom': ['English', 'en'],
    'Vietnam': ['Vietnamese', 'vi'],
    'Malaysia': ['Malay', 'ms'],
    'Chile': ['Spanish', 'es'],
    'Zimbabwe': ['English', 'en'],
    'Egypt': ['Arabic', 'ar'],
    'Japan': ['Japanese', 'ja'],
    'Iran': ['Persian', 'fa'],
    'Philippines': ['Filipino', 'fil'],
    'Israel': ['Hebrew', 'he'],
    'Bangladesh': ['Bengali', 'bn'],
    'Lebanon': ['Arabic', 'ar'],
    'Ukraine': ['Ukrainian', 'uk'],
    'United States': ['English', 'en'],
    'Czech Republic': ['Czech', 'cs'],
    'Italy': ['Italian', 'it'],
    'Indonesia': ['Indonesian', 'id'],
    'France': ['French', 'fr'],
    'Mexico': ['Spanish', 'es'],
    'Saudi Arabia': ['Arabic', 'ar'],
    'Nepal': ['Nepali', 'ne'],
    'Romania': ['Romanian', 'ro'],
    'Germany': ['German', 'de'],
    'Pakistan': ['Urdu', 'ur'],
    'Brazil': ['Portuguese', 'pt'],
    'Russia': ['Russian', 'ru'],
    'Taiwan': ['Mandarin Chinese', 'zh'],
    'Hong Kong': ['Cantonese', 'zh-yue'],
    'Peru': ['Spanish', 'es'],
    'Singapore': ['English', 'en'],
    'Turkey': ['Turkish', 'tr'],
    'New Zealand': ['English', 'en'],
    'Poland': ['Polish', 'pl'],
    'Argentina': ['Spanish', 'es'],
    'China': ['Mandarin Chinese', 'zh'],
    'South Africa': ['English', 'en'],
    'Thailand': ['Thai', 'th'],
    'Australia': ['English', 'en'],
    'Canada': ['English', 'en']
}