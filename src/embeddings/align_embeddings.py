import os 
import subprocess
from tqdm import tqdm
import sys
import gc

sys.path = ['/scratch/gpfs/vv7118/projects/MUSE']
emb_dim = 100
n_refinement = 1

script = """python ../MUSE/supervised.py --src_lang {lang1} --tgt_lang {lang2} \
--src_emb {src_emb} \
--tgt_emb {tgt_emb} \
--n_refinement {n_refinement} --dico_train default --emb_dim={emb_dim} \
--exp_path={exp_path} \
--exp_name={exp_name} \
"""

mapping = {
    'synthetic': '/scratch/gpfs/vv7118/projects/semantic-accents/embeddings/synthetic/Meta-Llama-3-1-70B-Instruct-htzs_{lang}/model.vec',
    'real': '/scratch/gpfs/vv7118/projects/semantic-accents/embeddings/real/human_{lang}/model.vec'
}

def main():
    langs = ['en', 'fr', 'et', 'hi', 'ja']
    settings = ['synthetic', 'real']

    for i, setting1 in enumerate(settings):
        for setting2 in settings[i:]:  # Only consider settings in the upper triangle, including the diagonal
            for j, lang1 in enumerate(tqdm(langs, desc="Processing languages for setting: " + setting1)):
                for lang2 in langs[j:]:  # Only consider languages after the current one

                    src_emb = mapping[setting1].format(lang=lang1)
                    tgt_emb = mapping[setting2].format(lang=lang2)
                    exp_path = f'/scratch/gpfs/vv7118/projects/semantic-accents/embeddings/aligned/{setting1}_{setting2}'
                    os.makedirs(exp_path, exist_ok=True)
                    exp_name = f'{lang1}_{lang2}'
                    # Run the subprocess
                    process = subprocess.run(
                        script.format(
                            lang1=lang1, lang2=lang2, 
                            n_refinement=n_refinement, 
                            src_emb=src_emb, tgt_emb=tgt_emb, 
                            emb_dim=emb_dim, exp_path=exp_path, exp_name=exp_name
                        ), 
                        shell=True
                    )

                    # Clean up the subprocess to release memory
                    del process
                    gc.collect()  # Force garbage collection to free memory

if __name__ == "__main__":
    main()