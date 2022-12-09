from harmony.generics import Sequential
from harmony.loader import PieceInfo, MetaCorporaInfo


def get_chord_transitions_by_era():
    metacorpora_path = 'petit_dcml_corpus/'
    metacorpora = MetaCorporaInfo.from_directory(metacorpora_path=metacorpora_path)

    # define piecewise operation
    local_key_segment_condition = lambda x: x[0].localkey == x[1].localkey

    piece_operation = lambda pieceinfo: [(x[0].chord_str, x[1].chord_str) for x in
                                         pieceinfo.get_tonal_harmony_sequential().get_n_grams(n=2).filter_by_condition(
                                             local_key_segment_condition)._seq]

    # define piece grouping
    eras = ['Renaissance', 'Baroque', 'Classical', 'Romantic']
    era_condition = lambda era: (lambda pieceinfo: pieceinfo.meta_info.era() == era)

    # automation
    transition_dict = {era: [piece_operation(piece) for piece in
                             metacorpora.filter_pieces_by_condition(era_condition(era))] for era in eras}

    return transition_dict


if __name__ == '__main__':
    # piece = PieceInfo.from_directory(parent_corpus_path='romantic_piano_corpus/debussy_suite_bergamasque/',
    #                                  piece_name='l075-03_suite_clair')

    #
    # bigrams_list = piece.get_tonal_harmony_sequential().get_n_grams(n=2).filter_by_condition(lambda x: x[0].localkey == x[1].localkey)
    # # print(bigrams_list)
    #
    # chords_str_bigrams = [(x[0].chord_str, x[1].chord_str) for x in bigrams_list._seq]
    # print(chords_str_bigrams)
    # print(len(chords_str_bigrams))

    result = get_chord_transitions_by_era()
    print(result)
