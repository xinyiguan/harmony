from dataclasses import dataclass
from typing import List, Literal, Tuple

import loader


@dataclass
class N_Gram:
    piece: loader.PieceInfo
    aspect: Literal['harmonies', 'measures', 'notes']
    key: str
    n: int

    def __post_init__(self):
        self.grams_seq = self.get_grams_seq()

    def get_grams_seq(self) -> List[str]:
        """
        produce a list of piece-wise grams as a list of tuples
        :return:
        """
        complete_seq = self.piece.get_aspect_df(aspect=self.aspect, selected_keys=[self.key]).values.flatten().tolist()
        grams_tuple = list(zip(*[complete_seq[i:] for i in range(self.n)]))
        grams = ['_'.join(grams_tuple[idx]) for idx, val in enumerate(grams_tuple)]
        return grams


if __name__ == '__main__':
    piece = loader.PieceInfo(parent_corpus_path='romantic_piano_corpus/debussy_suite_bergamasque/',
                             piece_name='l075-01_suite_prelude')
    unigram = piece.get_aspect_df(aspect='harmonies', selected_keys=['numeral']).values.flatten().tolist()
    n_gram = N_Gram(piece=piece, aspect='harmonies', key='numeral', n=2)
    print(len(n_gram.get_grams_seq()))
