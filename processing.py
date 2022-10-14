import pandas as pd

import old_loader


def get_corpus_plot_data_csv(corpus_path: str, write_csv: bool = False):
    corpus = loader.CorpusInfo(corpus_path)
    corpus_df = corpus.corpus_harmonies_meta_df

    corpus_name=corpus_df['corpus']
    fnames=corpus_df['fnames']
    composed_end=corpus_df['composed_end']
    annotated_key=corpus_df['annotated_key']
    chord = corpus_df['chord']
    numeral=corpus_df['numeral']
    chord_type =corpus_df['chord_type']

    corpus_plot_data=pd.concat([corpus_name, fnames, composed_end, annotated_key,
                               chord, numeral, chord_type], axis=1 )

    if write_csv:
        csv_file_name = str(corpus_name[0])+'_plot_data.csv'
        corpus_plot_data.to_csv(csv_file_name, sep='\t')
    return corpus_plot_data

if __name__ == '__main__':
    # corpus = loader.CorpusInfo('romantic_piano_corpus/debussy_suite_bergamasque/')
    # df = corpus.corpus_harmonies_meta_df
    # print(type(df['annotated_key']))

    corpus_path='romantic_piano_corpus/chopin_mazurkas/'
    get_corpus_plot_data_csv(corpus_path, write_csv=True)