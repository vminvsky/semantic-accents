

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