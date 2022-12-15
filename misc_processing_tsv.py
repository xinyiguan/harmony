#  By Xinyi Guan on 15 December 2022.

import os
import pandas as pd

if __name__ == '__main__':

    corpus_harmonnies_path = 'petit_dcml_corpus/pleyel_quartets/harmonies/'
    files = os.listdir(corpus_harmonnies_path)
    for file in files:
        # Open the file using pandas
        df = pd.read_csv(corpus_harmonnies_path+file, sep='\t')

        # Edit the header of the file
        df = df.applymap(lambda x: x.replace(' ', '') if isinstance(x, str) else x)
        df.rename(columns={col: col.strip() for col in df.columns}, inplace=True)

        # Save the edited file
        df.to_csv(corpus_harmonnies_path+file, sep='\t', index=False)
