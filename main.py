# Created by Xinyi Guan in 2022.

from musana.loader import MetaCorporaInfo


def get_chord_transitions_by_era():
    # define metacorpora
    metacorpora_path = 'petit_dcml_corpus/'
    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    # define piecewise operation
    piece_operation = lambda pieceinfo: pieceinfo.harmony_info.get_aspect(key='chord').n_grams(2)

    # define piece grouping
    eras = ['Renaissance', 'Baroque', 'Classical', 'Romantic']
    era_condition = lambda era: (lambda pieceinfo: pieceinfo.meta_info.era() == era)

    # automation
    transition_dict = {era: [piece_operation(piece) for piece in
                             metacorpora.filter_pieces_by_condition(era_condition(era))] for era in eras}

    print(transition_dict)
    return transition_dict


if __name__ == '__main__':
    # define metacorpora
    metacorpora_path = 'petit_dcml_corpus/'
    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    # define piecewise operation
    piece_operation = lambda pieceinfo: pieceinfo.chord_ngrams(n=2)
