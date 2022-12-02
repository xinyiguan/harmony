# Created by Xinyi Guan in 2022.
from harmony.loader import MetaCorpraInfo, CorpusInfo


if __name__=='__main__':
    meta_corpora_path = 'dcml_corpora/'
    metacopora = MetaCorpraInfo(meta_corpora_path)

    list_of_corpus = metacopora.corpus_name_list
    pieces_num_in_metacropora = []
    for item in list_of_corpus:
        corpus = CorpusInfo(meta_corpora_path+item+'/')
        pieces_num_in_corpus = len(corpus.get_annotated_piece_name_list())
        pieces_num_in_metacropora.append(pieces_num_in_corpus)

    num = sum(pieces_num_in_metacropora)
    print(pieces_num_in_metacropora)
    print(num)